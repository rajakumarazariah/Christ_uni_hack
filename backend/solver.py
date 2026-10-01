"""backend/solver.py
Deterministic resource-to-incident allocation engine using SciPy's Hungarian algorithm
(scipy.optimize.linear_sum_assignment).

Minimizes expected harm across all incidents, incorporates switch penalties to avoid
dispatch thrashing, and inserts dummy uncovered rows to handle resource deficits gracefully.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.config import Settings
from backend.harm_curves import harm, max_harm
from backend.state import (
    Assignment,
    Incident,
    IncidentStatus,
    Plan,
    Uncovered,
    Unit,
    UnitStatus,
    UnitType,
)

# Cost for incompatible unit-to-slot matches
BIG_COST = 1e6


def compute_incident_weight(severity: int, people_affected: int) -> float:
    """Computes the scaling weight for an incident based on severity and casualties.

    Formula: (severity / 5) * (1 + log10(max(people_affected, 1)) * 0.25)
    """
    safe_severity = max(1, min(5, severity))
    safe_people = max(1, people_affected)
    return (safe_severity / 5.0) * (1.0 + math.log10(safe_people) * 0.25)


def solve(
    incidents: Sequence[Incident],
    units: Sequence[Unit],
    eta_matrix: np.ndarray,
    previous_plan: Optional[Plan],
    settings: Settings,
) -> Plan:
    """Computes an optimal, deterministic assignment plan minimizing total expected harm.

    Args:
        incidents: All active incidents.
        units: All registered units (filtered internally for eligibility).
        eta_matrix: Matrix of shape [len(units) x len(incidents)] containing travel times in minutes.
        previous_plan: Previous committed or proposed plan (used for switch penalties and versioning).
        settings: Global application settings dataclass.

    Returns:
        A deterministic Plan instance containing unit assignments and uncovered incident slots.
    """
    # 1. Eligible units: status 'available' or 'en_route', sorted deterministically by id
    eligible_units_with_idx = [
        (idx, u)
        for idx, u in enumerate(units)
        if u.status in (UnitStatus.available, UnitStatus.en_route)
    ]
    eligible_units_with_idx.sort(key=lambda item: item[1].id)

    num_eligible_units = len(eligible_units_with_idx)

    # Map incident id to column index in eta_matrix
    incident_col_map: Dict[str, int] = {inc.id: idx for idx, inc in enumerate(incidents)}

    # 2. Expand non-resolved incidents into demand slots: (incident_id, unit_type, slot_index)
    slots: List[Tuple[Incident, UnitType, int]] = []
    # Sort incidents deterministically by id
    sorted_incidents = sorted(
        [inc for inc in incidents if inc.status != IncidentStatus.resolved],
        key=lambda x: x.id,
    )

    for inc in sorted_incidents:
        # Sort unit types for strict determinism
        for u_type in sorted(inc.units_needed.keys(), key=lambda t: t.value):
            count = inc.units_needed[u_type]
            for slot_idx in range(count):
                slots.append((inc, u_type, slot_idx))

    num_slots = len(slots)

    # Edge cases: no slots or no eligible units
    new_version = (previous_plan.version + 1) if previous_plan else 1
    plan_id = f"PLAN-{new_version}"

    if num_slots == 0:
        return Plan(
            id=plan_id,
            version=new_version,
            assignments=[],
            uncovered=[],
            total_harm=0.0,
            trigger="",
            status="proposed",
        )

    # 3 & 4. Build Cost Matrix
    # Matrix dimensions: Rows = (Real Units + Dummy Uncovered Rows), Cols = Slots
    # Dummy rows equal the number of slots so any slot can be left uncovered if beneficial.
    total_rows = num_eligible_units + num_slots
    cost_matrix = np.full((total_rows, num_slots), fill_value=BIG_COST, dtype=float)

    # Cache pre-calculated weights and max_harm values per incident
    inc_weights: Dict[str, float] = {
        inc.id: compute_incident_weight(inc.severity, inc.people_affected)
        for inc, _, _ in slots
    }
    inc_max_harms: Dict[str, float] = {
        inc.id: max_harm(inc.type, horizon=60.0) for inc, _, _ in slots
    }

    # Populate real unit rows (indices: 0 .. num_eligible_units - 1)
    for r_idx, (orig_unit_idx, unit) in enumerate(eligible_units_with_idx):
        for c_idx, (inc, slot_type, _) in enumerate(slots):
            if unit.type != slot_type:
                cost_matrix[r_idx, c_idx] = BIG_COST
                continue

            eta_col = incident_col_map.get(inc.id, 0)
            eta = float(eta_matrix[orig_unit_idx, eta_col])
            raw_harm = harm(inc.type, eta)
            weight = inc_weights[inc.id]
            cell_cost = weight * raw_harm

            # Switch penalty if unit is currently en_route to a DIFFERENT incident
            if (
                unit.status == UnitStatus.en_route
                and unit.assigned_incident_id is not None
                and unit.assigned_incident_id != inc.id
            ):
                cell_cost += settings.switch_penalty

            cost_matrix[r_idx, c_idx] = cell_cost

    # Populate dummy rows (indices: num_eligible_units .. total_rows - 1)
    # Dummy row d corresponds to leaving slot d uncovered
    for d_idx in range(num_slots):
        row_pos = num_eligible_units + d_idx
        inc, _, _ = slots[d_idx]
        dummy_cost = inc_weights[inc.id] * inc_max_harms[inc.id]
        cost_matrix[row_pos, d_idx] = dummy_cost

    # 5. Run SciPy Hungarian Solver (modified Jonker-Volgenant algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: List[Assignment] = []
    uncovered_list: List[Uncovered] = []
    total_harm_acc = 0.0

    # Count uncovered slots by (incident_id, unit_type)
    uncovered_counts: Dict[Tuple[str, UnitType], int] = {}

    for r, c in zip(row_ind, col_ind):
        inc, slot_type, _ = slots[c]
        chosen_cost = cost_matrix[r, c]

        if r < num_eligible_units and chosen_cost < (BIG_COST / 2.0):
            # Real unit matched to slot
            orig_unit_idx, unit = eligible_units_with_idx[r]
            eta_col = incident_col_map.get(inc.id, 0)
            eta = float(eta_matrix[orig_unit_idx, eta_col])
            raw_harm = harm(inc.type, eta)
            weight = inc_weights[inc.id]

            assignments.append(
                Assignment(
                    unit_id=unit.id,
                    incident_id=inc.id,
                    eta_min=round(eta, 2),
                    harm=round(raw_harm, 4),
                )
            )
            total_harm_acc += weight * raw_harm
        else:
            # Slot matched to a dummy row or infeasible unit -> Mark uncovered
            key = (inc.id, slot_type)
            uncovered_counts[key] = uncovered_counts.get(key, 0) + 1
            total_harm_acc += inc_weights[inc.id] * inc_max_harms[inc.id]

    # Build Uncovered models deterministically
    for (inc_id, u_type), count in sorted(uncovered_counts.items(), key=lambda x: (x[0][0], x[0][1].value)):
        uncovered_list.append(
            Uncovered(incident_id=inc_id, unit_type=u_type, missing=count)
        )

    # Sort assignments by unit_id for stable output
    assignments.sort(key=lambda a: a.unit_id)

    # 6. Return plan (trigger will be attached by orchestrator graph)
    return Plan(
        id=plan_id,
        version=new_version,
        assignments=assignments,
        uncovered=uncovered_list,
        total_harm=round(total_harm_acc, 4),
        trigger="",
        status="proposed",
    )