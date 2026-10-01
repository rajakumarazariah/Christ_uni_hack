"""backend/agents/graph.py
LangGraph workflow orchestrating Crisis Command's multi-agent emergency response.
Flow: intake -> logistics -> solver -> command -> approval_gate -> finalize.
Uses MemorySaver checkpointer and langgraph.types.interrupt to pause execution
for operator sign-off on high-risk diversions or severe unserviced incidents.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from backend.logistics import fetch_street_route
from backend.agents.command import evaluate_command_plan
from backend.agents.intake import extract
from backend.config import Settings, settings
from backend.logistics import build_eta_matrix
from backend.solver import solve
from backend.state import (
    Incident,
    IncidentStatus,
    Plan,
    PlanDiff,
    ReasoningCard,
    Unit,
    UnitStatus,
    store,
)

logger = logging.getLogger("crisis_command.graph")


# ---------------------------------------------------------------------------
# Workflow State Definition
# ---------------------------------------------------------------------------

class AgentWorkflowState(TypedDict, total=False):
    trigger: Dict[str, Any]             # {"kind": "new_incident"|"unit_down"|"hazard"|"manual", "payload": ...}
    incidents: List[Incident]
    units: List[Unit]
    eta_matrix: Any                     # np.ndarray
    meta: List[List[Dict[str, Any]]]
    proposed_plan: Optional[Plan]
    diff: Optional[PlanDiff]
    cards: List[ReasoningCard]
    flags: List[Dict[str, Any]]
    approved: Optional[bool]
    plan_id: str
    staging_suggestions: List[Dict[str, Any]]
    coverage: Dict[str, Any]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def intake_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Extracts structured incident data if unstructured, or incorporates raw incidents."""
    trigger = state.get("trigger", {})
    kind = trigger.get("kind", "manual")
    payload = trigger.get("payload", {})

    store.log_event("intake", f"Ingesting trigger: {kind}")

    if kind == "new_incident":
        # Check if already structured
        if isinstance(payload, Incident):
            inc_obj = payload
            store.add_incident(inc_obj)
        elif isinstance(payload, dict) and "type" in payload and "lat" in payload and payload["lat"] is not None:
            # Direct structured creation (e.g. from map click)
            if not payload.get("id"):
                payload["id"] = store.next_incident_id()
            inc_obj = Incident(**payload)
            store.add_incident(inc_obj)
        else:
            # Unstructured text or audio -> Run extraction agent
            text = payload.get("text") if isinstance(payload, dict) else str(payload)
            audio = payload.get("audio_bytes") if isinstance(payload, dict) else None
            mime = payload.get("mime_type") if isinstance(payload, dict) else None

            res = await extract(text=text, audio_bytes=audio, mime_type=mime)

            if res.lat is not None and res.lng is not None and res.incident_type is not None:
                inc_obj = Incident(
                    id=store.next_incident_id(),
                    type=res.incident_type,
                    severity=res.severity or 3,
                    people_affected=res.people_affected or 1,
                    lat=res.lat,
                    lng=res.lng,
                    location_text=res.location_text,
                    confidence=res.confidence,
                    needs_confirmation=res.needs_confirmation,
                    source="voice" if audio else "text",
                )
                store.add_incident(inc_obj)
            else:
                store.log_event("intake", f"Extraction incomplete: coords unresolved for '{res.location_text}'")

    elif kind == "unit_down":
        unit_id = payload.get("unit_id") if isinstance(payload, dict) else str(payload)
        store.set_unit_status(unit_id, UnitStatus.down)
        store.log_event("system", f"Unit {unit_id} marked DOWN.")

    elif kind == "hazard":
        if isinstance(payload, dict):
            if not payload.get("id"):
                payload["id"] = store.next_hazard_id()
            from backend.state import Hazard
            store.add_hazard(Hazard(**payload))

    # Pull current active state
    active_incidents = [
        inc for inc in store.incidents.values()
        if inc.status != IncidentStatus.resolved
    ]
    all_units = list(store.units.values())

    return {
        "incidents": active_incidents,
        "units": all_units,
    }


async def logistics_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Calculates updated ETA travel time matrix with hazard avoidance."""
    incidents = state.get("incidents", [])
    units = state.get("units", [])
    hazards = list(store.hazards.values())

    store.log_event("logistics", f"Building travel time matrix for {len(units)} units and {len(incidents)} incidents.")
    matrix, meta = await build_eta_matrix(units, incidents, hazards)

    return {
        "eta_matrix": matrix,
        "meta": meta,
    }


async def solver_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Executes the SciPy Hungarian optimization algorithm."""
    incidents = state.get("incidents", [])
    units = state.get("units", [])
    eta_matrix = state.get("eta_matrix")
    trigger = state.get("trigger", {})

    store.log_event("solver", "Executing Hungarian allocation minimizing expected harm.")

    if eta_matrix is None:
        eta_matrix = np.zeros((len(units), len(incidents)), dtype=float)

    plan = solve(
        incidents=incidents,
        units=units,
        eta_matrix=eta_matrix,
        previous_plan=store.current_plan,
        settings=settings,
    )
    plan.trigger = trigger.get("kind", "manual")
    store.pending_plan = plan

    for assign in plan.assignments:
        unit = next((u for u in units if u.id == assign.unit_id), None)
        inc = next((i for i in incidents if i.id == assign.incident_id), None)
        if unit and inc:
            assign.route_geometry = await fetch_street_route(unit.lat, unit.lng, inc.lat, inc.lng)

    return {
        "proposed_plan": plan,
        "plan_id": plan.id,
    }



async def command_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Computes plan diffs, reasoning cards, risk flags, and coverage analytics."""
    proposed_plan = state["proposed_plan"]
    incidents = state.get("incidents", [])
    units = state.get("units", [])

    store.log_event("command", "Evaluating plan diff, generating operator reasoning, and scanning risk flags.")

    eval_result = await evaluate_command_plan(
        old_plan=store.current_plan,
        new_plan=proposed_plan,
        incidents=incidents,
        units=units,
    )

    return {
        "diff": eval_result["diff"],
        "cards": eval_result["cards"],
        "flags": eval_result["approval_flags"],
        "staging_suggestions": eval_result["staging_suggestions"],
        "coverage": eval_result["coverage"],
    }


def approval_gate_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Checks for risk conditions.

    If present, interrupts workflow execution and yields control to the human dispatcher.
    Otherwise auto-approves.
    """
    flags = state.get("flags", [])
    plan = state["proposed_plan"]

    if flags:
        store.log_event("command", f"Plan {plan.id} flagged with {len(flags)} risk items. Awaiting dispatcher review.")
        # Pause execution and return approval payload to caller
        approval_decision = interrupt({
            "message": "Operator sign-off required for flagged high-risk actions.",
            "plan_id": plan.id,
            "flags": flags,
            "diff": state.get("diff").model_dump() if state.get("diff") else None,
            "cards": [c.model_dump() for c in state.get("cards", [])],
        })
        # Resumed via resolve_approval with a boolean value
        return {"approved": bool(approval_decision)}

    store.log_event("command", f"Plan {plan.id} carries standard risk profile. Auto-approving.")
    return {"approved": True}


async def finalize_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """Finalizes dispatch decisions.

    If approved: commits the proposed plan to state.
    If rejected: runs a conservative fallback solve (high switch penalty) to eliminate unit diversions.
    """
    approved = state.get("approved", True)
    proposed_plan = state["proposed_plan"]

    if approved:
        store.log_event("human", f"Plan {proposed_plan.id} APPROVED by command.")
        store.commit_plan(proposed_plan)
        return {"proposed_plan": proposed_plan, "approved": True}

    store.log_event("human", f"Plan {proposed_plan.id} REJECTED by command. Generating conservative fallback plan.")

    # Build safe fallback: run solver with a prohibitive switch penalty to stop diversions
    safe_settings = Settings(
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        llm_mock=settings.llm_mock,
        use_osrm=settings.use_osrm,
        osrm_base_url=settings.osrm_base_url,
        nominatim_url=settings.nominatim_url,
        nominatim_user_agent=settings.nominatim_user_agent,
        demo_city_name=settings.demo_city_name,
        demo_center_lat=settings.demo_center_lat,
        demo_center_lng=settings.demo_center_lng,
        demo_bbox=settings.demo_bbox,
        target_response_min=settings.target_response_min,
        switch_penalty=1e5,  # Prohibit switching active en-route units
        avg_speed_kmph=settings.avg_speed_kmph,
        sim_speed=settings.sim_speed,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )

    incidents = state.get("incidents", [])
    units = state.get("units", [])
    eta_matrix = state.get("eta_matrix")

    if eta_matrix is None:
        eta_matrix = np.zeros((len(units), len(incidents)), dtype=float)

    safe_plan = solve(
        incidents=incidents,
        units=units,
        eta_matrix=eta_matrix,
        previous_plan=store.current_plan,
        settings=safe_settings,
    )
    safe_plan.trigger = "rejection_safe_fallback"
    store.log_event("solver", f"Conservative fallback plan {safe_plan.id} calculated without diversions.")
    store.commit_plan(safe_plan)

    return {"proposed_plan": safe_plan, "approved": False}


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def _build_workflow():
    workflow = StateGraph(AgentWorkflowState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("logistics", logistics_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("command", command_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "logistics")
    workflow.add_edge("logistics", "solver")
    workflow.add_edge("solver", "command")
    workflow.add_edge("command", "approval_gate")
    workflow.add_edge("approval_gate", "finalize")
    workflow.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Compiled LangGraph application instance
graph_app = _build_workflow()


# ---------------------------------------------------------------------------
# Public Execution APIs
# ---------------------------------------------------------------------------

async def replan(trigger: Dict[str, Any]) -> Dict[str, Any]:
    """Initiates an end-to-end planning cycle triggered by a state change.

    If risky actions require approval, execution halts at the approval_gate,
    returning the proposal with interruption details.
    """
    initial_plan_id = f"PLAN-{(store.current_plan.version + 1) if store.current_plan else 1}"
    config = {"configurable": {"thread_id": initial_plan_id}}

    initial_state: AgentWorkflowState = {
        "trigger": trigger,
        "plan_id": initial_plan_id,
    }

    # Execute workflow until completion or human approval interrupt
    output_state = await graph_app.ainvoke(initial_state, config=config)

    # Inspect if execution stopped at an interrupt
    graph_state = await graph_app.aget_state(config)
    is_interrupted = bool(graph_state.tasks and any(task.interrupts for task in graph_state.tasks))

    current_data = graph_state.values

    return {
        "thread_id": initial_plan_id,
        "interrupted": is_interrupted,
        "proposed_plan": (
            current_data["proposed_plan"].model_dump()
            if current_data.get("proposed_plan") else None
        ),
        "diff": current_data["diff"].model_dump() if current_data.get("diff") else None,
        "cards": [c.model_dump() for c in current_data.get("cards", [])],
        "flags": current_data.get("flags", []),
        "approved": current_data.get("approved"),
        "coverage": current_data.get("coverage"),
        "staging_suggestions": current_data.get("staging_suggestions"),
    }


async def resolve_approval(thread_id: str, approve: bool) -> Dict[str, Any]:
    """Resumes an interrupted workflow with the operator's decision (True to approve, False to reject)."""
    config = {"configurable": {"thread_id": thread_id}}

    # Resume graph execution passing the approval decision to the interrupt site
    output_state = await graph_app.ainvoke(Command(resume=approve), config=config)

    committed = store.current_plan.model_dump() if store.current_plan else None

    return {
        "thread_id": thread_id,
        "approved": approve,
        "status": "committed" if approve else "fallback_committed",
        "current_plan": committed,
    }