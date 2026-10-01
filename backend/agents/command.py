"""backend/agents/command.py
Command reasoning agent for Crisis Command:
- Pure deterministic computation of plan diffs and numeric impact cards.
- Human-in-the-loop approval flagging for high-stakes operational risks.
- Grounded Gemini pass for natural-language card narration (strictly preserving numbers).
- Integration of logistics coverage analysis and proactive staging suggestions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from google import genai
from pydantic import BaseModel, Field

from backend.config import settings
from backend.harm_curves import max_harm
from backend.logistics import coverage_gaps, suggest_staging
from backend.state import (
    Assignment,
    DiffChange,
    Incident,
    Plan,
    PlanDiff,
    ReasoningCard,
    Unit,
    UnitStatus,
)

logger = logging.getLogger("crisis_command.command")


# ---------------------------------------------------------------------------
# 1. Compute Plan Diff
# ---------------------------------------------------------------------------

def compute_diff(old_plan: Optional[Plan], new_plan: Plan) -> PlanDiff:
    """Computes deterministic delta between two dispatch plans keyed by unit_id."""
    old_assigns: Dict[str, Assignment] = (
        {a.unit_id: a for a in old_plan.assignments} if old_plan else {}
    )
    new_assigns: Dict[str, Assignment] = {a.unit_id: a for a in new_plan.assignments}

    added: List[Assignment] = []
    removed: List[Assignment] = []
    changed: List[DiffChange] = []
    unchanged_count = 0

    all_unit_ids = sorted(set(old_assigns.keys()) | set(new_assigns.keys()))

    for u_id in all_unit_ids:
        in_old = u_id in old_assigns
        in_new = u_id in new_assigns

        if in_new and not in_old:
            added.append(new_assigns[u_id])
        elif in_old and not in_new:
            removed.append(old_assigns[u_id])
        else:
            old_a = old_assigns[u_id]
            new_a = new_assigns[u_id]
            if old_a.incident_id != new_a.incident_id:
                changed.append(
                    DiffChange(
                        unit_id=u_id,
                        from_incident_id=old_a.incident_id,
                        to_incident_id=new_a.incident_id,
                        eta_before=old_a.eta_min,
                        eta_after=new_a.eta_min,
                        harm_delta=round(new_a.harm - old_a.harm, 4),
                    )
                )
            else:
                unchanged_count += 1

    return PlanDiff(
        added=added,
        removed=removed,
        changed=changed,
        unchanged_count=unchanged_count,
    )


# ---------------------------------------------------------------------------
# 2. Build Numeric Reasoning Cards
# ---------------------------------------------------------------------------

def build_cards(
    diff: PlanDiff,
    incidents: Sequence[Incident],
    units: Sequence[Unit],
    old_plan: Optional[Plan],
    new_plan: Plan,
) -> List[ReasoningCard]:
    """Generates structured reasoning cards for each changed unit.

    All numerical fields (ETA, harm deltas, abandonment delays) are computed
    deterministically from the state models; the LLM is never allowed to fabricate numbers.
    """
    inc_map = {inc.id: inc for inc in incidents}
    unit_map = {u.id: u for u in units}
    cards: List[ReasoningCard] = []

    for chg in diff.changed:
        from_inc = inc_map.get(chg.from_incident_id or "")
        to_inc = inc_map.get(chg.to_incident_id)
        unit = unit_map.get(chg.unit_id)

        unit_name = unit.name if unit else chg.unit_id
        from_name = from_inc.location_text if from_inc else (chg.from_incident_id or "prior location")
        to_name = to_inc.location_text if to_inc else chg.to_incident_id

        # Delay penalty added to the abandoned incident
        abandoned_harm = max_harm(from_inc.type, horizon=60.0) if from_inc else 0.5
        eta_before = chg.eta_before or 0.0

        numbers: Dict[str, Any] = {
            "eta_before": round(eta_before, 1),
            "eta_after": round(chg.eta_after, 1),
            "eta_diff_min": round(chg.eta_after - eta_before, 1),
            "harm_delta": chg.harm_delta,
            "abandoned_incident_id": chg.from_incident_id,
            "abandoned_incident_uncovered_harm": round(abandoned_harm, 3),
            "target_severity": to_inc.severity if to_inc else 3,
            "previous_severity": from_inc.severity if from_inc else 1,
        }

        # Safe template string fallback
        template_text = (
            f"Divert {unit_name} from {from_name} ({chg.from_incident_id}) to {to_name} ({chg.to_incident_id}). "
            f"ETA changes from {eta_before:.1f}m to {chg.eta_after:.1f}m "
            f"to mitigate higher-severity hazard."
        )

        cards.append(
            ReasoningCard(
                unit_id=chg.unit_id,
                from_incident_id=chg.from_incident_id,
                to_incident_id=chg.to_incident_id,
                text=template_text,
                numbers=numbers,
                needs_approval=False,  # Evaluated by flag_approvals
            )
        )

    return cards


# ---------------------------------------------------------------------------
# 3. Flag Risky Approvals
# ---------------------------------------------------------------------------

def flag_approvals(
    diff: PlanDiff,
    new_plan: Plan,
    incidents: Sequence[Incident],
    units: Sequence[Unit],
) -> List[Dict[str, Any]]:
    """Identifies high-risk decisions requiring human dispatcher sign-off.

    Triggers human gate when:
    (a) An active en-route unit is diverted away from its assigned incident.
    (b) Any incident with severity >= 4 is uncovered or under-covered.
    (c) An assignment serves an incident marked with needs_confirmation.
    """
    unit_map = {u.id: u for u in units}
    inc_map = {inc.id: inc for inc in incidents}
    flags: List[Dict[str, Any]] = []

    # Condition (a): Diverting an active en_route unit
    for chg in diff.changed:
        unit = unit_map.get(chg.unit_id)
        if unit and unit.status == UnitStatus.en_route:
            flags.append({
                "type": "unit_diverted",
                "unit_id": unit.id,
                "from_incident_id": chg.from_incident_id,
                "to_incident_id": chg.to_incident_id,
                "reason": f"Unit {unit.name} ({unit.id}) is actively en route and would be diverted.",
            })

    # Condition (b): High-severity (>= 4) incident left uncovered
    for unc in new_plan.uncovered:
        inc = inc_map.get(unc.incident_id)
        if inc and inc.severity >= 4:
            flags.append({
                "type": "high_severity_uncovered",
                "incident_id": inc.id,
                "severity": inc.severity,
                "unit_type": unc.unit_type.value,
                "missing": unc.missing,
                "reason": (
                    f"Priority incident {inc.id} ({inc.type.value}, severity {inc.severity}) "
                    f"remains deficient of {unc.missing} {unc.unit_type.value}(s)."
                ),
            })

    # Condition (c): Assignment depends on unverified incident
    for assign in new_plan.assignments:
        inc = inc_map.get(assign.incident_id)
        if inc and inc.needs_confirmation:
            flags.append({
                "type": "unconfirmed_incident_assignment",
                "unit_id": assign.unit_id,
                "incident_id": inc.id,
                "reason": f"Assignment to {inc.id} relies on low-confidence or unconfirmed triage details.",
            })

    return flags


# ---------------------------------------------------------------------------
# 4. LLM Narration Pass
# ---------------------------------------------------------------------------

class NarrationResponse(BaseModel):
    narratives: List[str] = Field(
        description="One fluent sentence per card. Must preserve all numbers and facts exactly."
    )


async def narrate(cards: List[ReasoningCard]) -> List[ReasoningCard]:
    """Uses Gemini to polish card descriptions into concise operator sentences.

    Strictly forbids altering, adding, or deleting numerical values. Falls back
    cleanly to deterministic template text if mock mode is on or an API error occurs.
    """
    if not cards:
        return []

    if settings.llm_mock or not settings.gemini_api_key:
        return cards

    facts = [
        f"Card {i}: Unit {c.unit_id} moves from {c.from_incident_id or 'standby'} to {c.to_incident_id}. "
        f"ETA before: {c.numbers.get('eta_before', 'N/A')}m, ETA after: {c.numbers.get('eta_after', 'N/A')}m, "
        f"Target Severity: {c.numbers.get('target_severity', 3)}."
        for i, c in enumerate(cards)
    ]

    prompt = (
        "You are an emergency command center AI assistant. "
        "Reword each dispatch item below into exactly ONE clear, authoritative operational sentence. "
        "CRITICAL RULE: Do NOT add, change, round, or alter ANY numbers or IDs. Use the exact numbers given.\n\n"
        + "\n".join(facts)
    )

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "response_schema": NarrationResponse,
                },
            ),
            timeout=8.0,
        )

        parsed: Optional[NarrationResponse] = resp.parsed  # type: ignore[assignment]
        if parsed and len(parsed.narratives) == len(cards):
            for i, c in enumerate(cards):
                c.text = parsed.narratives[i].strip()
    except Exception as exc:
        logger.warning("LLM narration failed; retaining deterministic text: %s", exc)

    return cards


# ---------------------------------------------------------------------------
# 5. Command Evaluation Package
# ---------------------------------------------------------------------------

async def evaluate_command_plan(
    old_plan: Optional[Plan],
    new_plan: Plan,
    incidents: Sequence[Incident],
    units: Sequence[Unit],
) -> Dict[str, Any]:
    """Orchestrates diff calculation, reasoning cards, human-gate checks,

    and spatial logistics analytics for UI streaming.
    """
    diff = compute_diff(old_plan, new_plan)
    cards = build_cards(diff, incidents, units, old_plan, new_plan)
    approval_flags = flag_approvals(diff, new_plan, incidents, units)

    # Attach approval requirement directly to matching cards
    diverted_units = {f["unit_id"] for f in approval_flags if f.get("unit_id")}
    for c in cards:
        if c.unit_id in diverted_units:
            c.needs_approval = True

    # Narrative pass
    narrated_cards = await narrate(cards)

    # Logistics coverage & staging evaluation
    gaps = coverage_gaps(units, settings.demo_bbox)
    staging = suggest_staging(units, gaps)

    return {
        "diff": diff,
        "cards": narrated_cards,
        "approval_flags": approval_flags,
        "requires_human_approval": len(approval_flags) > 0,
        "coverage": gaps,
        "staging_suggestions": staging,
    }