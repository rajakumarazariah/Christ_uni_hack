"""tests/test_solver.py
Pytest suite verifying the deterministic properties, triage hierarchy,
switch penalty mechanics, and uncovered deficit handling of backend/solver.py.
"""

import numpy as np
import pytest

from backend.config import Settings
from backend.state import (
    Incident,
    IncidentType,
    Plan,
    Unit,
    UnitStatus,
    UnitType,
)
from backend.solver import solve


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        gemini_api_key="",
        gemini_model="gemini-2.5-flash",
        llm_mock=True,
        use_osrm=False,
        osrm_base_url="https://router.project-osrm.org",
        nominatim_url="https://nominatim.openstreetmap.org",
        nominatim_user_agent="test-agent",
        demo_city_name="Tiruchirappalli",
        demo_center_lat=10.8050,
        demo_center_lng=78.6856,
        demo_bbox=(10.72, 78.60, 10.90, 78.78),
        target_response_min=10,
        switch_penalty=0.15,
        avg_speed_kmph=40.0,
        sim_speed=30,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


def test_triage_cardiac_over_minor_injury(mock_settings: Settings) -> None:
    """One ambulance, a cardiac arrest 6 km away (~9 min) vs a minor injury 1 km away (~1.5 min).

    Despite the distance disadvantage, the cardiac case's severe harm curve and severity 5
    must win the assignment over the minor injury.
    """
    ambulance = Unit(
        id="AMB-1",
        type=UnitType.ambulance,
        name="Ambulance 1",
        lat=10.8000,
        lng=78.6800,
        home_lat=10.8000,
        home_lng=78.6800,
        status=UnitStatus.available,
    )

    inc_cardiac = Incident(
        id="INC-CARDIAC",
        type=IncidentType.cardiac_arrest,
        severity=5,
        people_affected=1,
        lat=10.8600,
        lng=78.6800,
        location_text="Far location",
        units_needed={UnitType.ambulance: 1},
    )

    inc_minor = Incident(
        id="INC-MINOR",
        type=IncidentType.minor_injury,
        severity=1,
        people_affected=1,
        lat=10.8100,
        lng=78.6800,
        location_text="Nearby location",
        units_needed={UnitType.ambulance: 1},  # Force ambulance need for test competition
    )

    incidents = [inc_cardiac, inc_minor]
    units = [ambulance]

    # Column 0: cardiac (~9 min), Column 1: minor (~1.5 min)
    eta_matrix = np.array([[9.0, 1.5]])

    plan = solve(incidents, units, eta_matrix, previous_plan=None, settings=mock_settings)

    assert len(plan.assignments) == 1
    assert plan.assignments[0].unit_id == "AMB-1"
    assert plan.assignments[0].incident_id == "INC-CARDIAC"

    # Verify minor injury is left uncovered
    assert len(plan.uncovered) == 1
    assert plan.uncovered[0].incident_id == "INC-MINOR"


def test_resource_deficit_leaves_lowest_harm_uncovered(mock_settings: Settings) -> None:
    """Two ambulances, three slots: the lowest-harm slot must be the one left uncovered."""
    amb1 = Unit(
        id="AMB-1",
        type=UnitType.ambulance,
        name="Ambulance 1",
        lat=10.8000,
        lng=78.6800,
        home_lat=10.8000,
        home_lng=78.6800,
        status=UnitStatus.available,
    )
    amb2 = Unit(
        id="AMB-2",
        type=UnitType.ambulance,
        name="Ambulance 2",
        lat=10.8100,
        lng=78.6800,
        home_lat=10.8100,
        home_lng=78.6800,
        status=UnitStatus.available,
    )

    inc_cardiac = Incident(
        id="INC-1",
        type=IncidentType.cardiac_arrest,
        severity=5,
        lat=10.8050,
        lng=78.6800,
        location_text="Central",
        units_needed={UnitType.ambulance: 1},
    )
    inc_trauma = Incident(
        id="INC-2",
        type=IncidentType.major_trauma,
        severity=4,
        lat=10.8080,
        lng=78.6800,
        location_text="North-Central",
        units_needed={UnitType.ambulance: 1},
    )
    inc_minor = Incident(
        id="INC-3",
        type=IncidentType.minor_injury,
        severity=1,
        lat=10.8020,
        lng=78.6800,
        location_text="South-Central",
        units_needed={UnitType.ambulance: 1},
    )

    units = [amb1, amb2]
    incidents = [inc_cardiac, inc_trauma, inc_minor]

    # Equal ETAs (5 min) across all units and incidents to isolate harm weighting
    eta_matrix = np.full((2, 3), 5.0)

    plan = solve(incidents, units, eta_matrix, previous_plan=None, settings=mock_settings)

    assigned_inc_ids = {a.incident_id for a in plan.assignments}
    assert "INC-1" in assigned_inc_ids
    assert "INC-2" in assigned_inc_ids
    assert "INC-3" not in assigned_inc_ids

    assert len(plan.uncovered) == 1
    assert plan.uncovered[0].incident_id == "INC-3"


def test_switch_penalty_prevents_marginal_swaps_but_permits_critical_reroutes(mock_settings: Settings) -> None:
    """An en-route unit is NOT switched for a marginal gain, but IS diverted for a critical severity-5 case."""
    # Scenario A: Unit is en_route to INC-A (Trauma, severity 3).
    # New incident INC-B appears (Trauma, severity 3) with an ETA that is 1 minute faster.
    # The marginal gain (~0.016 harm) is smaller than switch_penalty (0.15), so it should NOT switch.
    amb_en_route = Unit(
        id="AMB-1",
        type=UnitType.ambulance,
        name="Ambulance 1",
        lat=10.8000,
        lng=78.6800,
        home_lat=10.8000,
        home_lng=78.6800,
        status=UnitStatus.en_route,
        assigned_incident_id="INC-A",
    )

    inc_a = Incident(
        id="INC-A",
        type=IncidentType.major_trauma,
        severity=3,
        lat=10.8200,
        lng=78.6800,
        location_text="Incident A",
        units_needed={UnitType.ambulance: 1},
    )
    inc_b = Incident(
        id="INC-B",
        type=IncidentType.major_trauma,
        severity=3,
        lat=10.8100,
        lng=78.6800,
        location_text="Incident B (Marginally closer)",
        units_needed={UnitType.ambulance: 1},
    )

    # ETA: INC-A = 5 min, INC-B = 4 min
    eta_matrix_marginal = np.array([[5.0, 4.0]])

    plan_marginal = solve(
        [inc_a, inc_b], [amb_en_route], eta_matrix_marginal, previous_plan=None, settings=mock_settings
    )

    assert len(plan_marginal.assignments) == 1
    # Remains committed to INC-A due to switch penalty
    assert plan_marginal.assignments[0].incident_id == "INC-A"

    # Scenario B: Unit is en_route to INC-A (Trauma, severity 3).
    # Critical INC-CRITICAL (Cardiac, severity 5) arrives.
    # The harm reduction vastly exceeds the 0.15 switch penalty, so the unit MUST be diverted.
    inc_critical = Incident(
        id="INC-CRITICAL",
        type=IncidentType.cardiac_arrest,
        severity=5,
        lat=10.8300,
        lng=78.6800,
        location_text="Cardiac Incident",
        units_needed={UnitType.ambulance: 1},
    )

    # ETA: INC-A = 5 min, INC-CRITICAL = 6 min
    eta_matrix_critical = np.array([[5.0, 3.0]])

    plan_critical = solve(
        [inc_a, inc_critical], [amb_en_route], eta_matrix_critical, previous_plan=None, settings=mock_settings
    )

    assert len(plan_critical.assignments) == 1
    # Unit successfully rerouted to save the cardiac patient
    assert plan_critical.assignments[0].incident_id == "INC-CRITICAL"


def test_solver_strict_determinism(mock_settings: Settings) -> None:
    """Executing the solver with identical inputs must always yield the exact same plan."""
    units = [
        Unit(
            id=f"AMB-{i}",
            type=UnitType.ambulance,
            name=f"Amb {i}",
            lat=10.8000 + (i * 0.01),
            lng=78.6800,
            home_lat=10.8000,
            home_lng=78.6800,
            status=UnitStatus.available,
        )
        for i in range(1, 5)
    ]

    incidents = [
        Incident(
            id=f"INC-{j}",
            type=IncidentType.major_trauma,
            severity=3,
            lat=10.8050,
            lng=78.6800 + (j * 0.01),
            location_text=f"Location {j}",
            units_needed={UnitType.ambulance: 1},
        )
        for j in range(1, 4)
    ]

    eta_matrix = np.array([
        [4.2, 8.1, 6.0],
        [5.0, 3.2, 7.5],
        [9.1, 4.4, 2.8],
        [6.2, 5.5, 4.1],
    ])

    plan1 = solve(incidents, units, eta_matrix, previous_plan=None, settings=mock_settings)
    plan2 = solve(incidents, units, eta_matrix, previous_plan=None, settings=mock_settings)

    assert plan1.total_harm == plan2.total_harm
    assert [a.model_dump() for a in plan1.assignments] == [a.model_dump() for a in plan2.assignments]
    assert [u.model_dump() for u in plan1.uncovered] == [u.model_dump() for u in plan2.uncovered]