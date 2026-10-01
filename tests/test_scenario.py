"""tests/test_scenario.py
End-to-end operational scenario test for Crisis Command.
Simulates opening demo triage, injecting a high-severity cardiac arrest,
marking responding units down, asserting plan diff calculations, reasoning cards,
human approval gating on en-route diversions, and strict assignment determinism.
Runs fully offline with LLM_MOCK=True and USE_OSRM=False.
"""

import pytest

from backend.agents.command import compute_diff, evaluate_command_plan
from backend.config import Settings
from backend.logistics import build_eta_matrix
from backend.solver import solve
from backend.state import (
    Incident,
    IncidentStatus,
    IncidentType,
    Plan,
    Unit,
    UnitStatus,
    UnitType,
    store,
)


@pytest.fixture
def test_settings() -> Settings:
    """Provides isolated offline settings for end-to-end integration tests."""
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


@pytest.mark.asyncio
async def test_full_incident_escalation_and_diversion_scenario(test_settings: Settings) -> None:
    # 1. Reset state store and load baseline demo scenario
    store.load_demo()
    assert len(store.units) == 11, "Expected 11 baseline units in resources.json"
    assert len(store.incidents) == 3, "Expected 3 initial seed incidents"

    active_incidents = list(store.incidents.values())
    all_units = list(store.units.values())

    # 2. Compute initial opening plan (v1)
    eta_matrix_v1, _ = await build_eta_matrix(all_units, active_incidents, hazards=[])
    plan_v1: Plan = solve(
        incidents=active_incidents,
        units=all_units,
        eta_matrix=eta_matrix_v1,
        previous_plan=None,
        settings=test_settings,
    )
    store.commit_plan(plan_v1)
    assert plan_v1.version == 1
    assert len(plan_v1.assignments) > 0

    # Ensure assigned units are marked en_route with assigned_incident_id
    assigned_v1 = {a.unit_id: a.incident_id for a in plan_v1.assignments}
    for u_id, inc_id in assigned_v1.items():
        assert store.units[u_id].status == UnitStatus.en_route
        assert store.units[u_id].assigned_incident_id == inc_id

    # 3. Inject new Critical Incident (Cardiac arrest, Severity 5)
    critical_incident = Incident(
        id=store.next_incident_id(),
        type=IncidentType.cardiac_arrest,
        severity=5,
        people_affected=1,
        lat=10.8050,
        lng=78.6870,
        location_text="Near Central Bus Stand Clock Tower",
        units_needed={UnitType.ambulance: 1},
        source="text",
        status=IncidentStatus.open,
    )
    store.add_incident(critical_incident)

    # 4. Mark one active responding ambulance as DOWN (mechanical failure)
    # Pick the first ambulance currently en_route from Plan v1
    en_route_ambs = [
        u for u in store.units.values()
        if u.type == UnitType.ambulance and u.status == UnitStatus.en_route
    ]
    assert len(en_route_ambs) > 0, "Expected at least one ambulance en route in opening plan"
    failing_ambulance = sorted(en_route_ambs, key=lambda u: u.id)[0]
    store.set_unit_status(failing_ambulance.id, UnitStatus.down)

    # 5. Solve for Plan v2 with the updated state
    current_incidents = [inc for inc in store.incidents.values() if inc.status != IncidentStatus.resolved]
    current_units = list(store.units.values())

    eta_matrix_v2, _ = await build_eta_matrix(current_units, current_incidents, hazards=[])
    plan_v2: Plan = solve(
        incidents=current_incidents,
        units=current_units,
        eta_matrix=eta_matrix_v2,
        previous_plan=store.current_plan,
        settings=test_settings,
    )

    # Assert: Down unit is NOT assigned anywhere in the new plan
    assigned_v2_units = {a.unit_id for a in plan_v2.assignments}
    assert failing_ambulance.id not in assigned_v2_units, f"Unit {failing_ambulance.id} is down and must not be assigned"

    # Assert: Plan v2 differs from Plan v1
    diff = compute_diff(plan_v1, plan_v2)
    plan_has_changed = len(diff.added) > 0 or len(diff.removed) > 0 or len(diff.changed) > 0
    assert plan_has_changed, "Plan v2 must reflect changes after injecting incident and disabling unit"

    # Evaluate Command intelligence: reasoning cards and approval risk flags
    eval_result = await evaluate_command_plan(
        old_plan=plan_v1,
        new_plan=plan_v2,
        incidents=current_incidents,
        units=current_units,
    )

    # Assert: A reasoning card exists for every diverted unit
    changed_unit_ids = {chg.unit_id for chg in diff.changed}
    card_unit_ids = {c.unit_id for c in eval_result["cards"]}
    assert changed_unit_ids == card_unit_ids, "Every changed/diverted unit must have an associated reasoning card"

    # Assert: If any unit actively en_route in v1 is diverted in v2, an approval flag is raised
    en_route_unit_ids = {
        u.id for u in current_units
        if u.status == UnitStatus.en_route and u.id != failing_ambulance.id
    }
    diverted_en_route_units = changed_unit_ids.intersection(en_route_unit_ids)

    diverted_flags = [
        f for f in eval_result["approval_flags"]
        if f.get("type") == "unit_diverted"
    ]

    if diverted_en_route_units:
        assert len(diverted_flags) > 0, "Approval flag must be raised when an active en_route unit is diverted"
        flagged_unit_ids = {f["unit_id"] for f in diverted_flags}
        for u_id in diverted_en_route_units:
            assert u_id in flagged_unit_ids

    # 6. Strict Determinism Assertion: Running with identical inputs yields identical assignments
    plan_v2_repeat: Plan = solve(
        incidents=current_incidents,
        units=current_units,
        eta_matrix=eta_matrix_v2,
        previous_plan=store.current_plan,
        settings=test_settings,
    )
    assert plan_v2.total_harm == plan_v2_repeat.total_harm
    assert [a.model_dump() for a in plan_v2.assignments] == [a.model_dump() for a in plan_v2_repeat.assignments]
    assert [u.model_dump() for u in plan_v2.uncovered] == [u.model_dump() for u in plan_v2_repeat.uncovered]