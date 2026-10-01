"""backend/main.py
FastAPI application entrypoint for Crisis Command.
Exposes endpoints for multimodal incident intake, manual dispatch overrides,
resource status toggles, human-in-the-loop approvals, METHANE reporting,
and real-time Server-Sent Events (SSE) broadcasting.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agents.graph import replan, resolve_approval
from backend.agents.intake import IntakeResult, extract
from backend.config import settings
from backend.logistics import coverage_gaps, suggest_staging
from backend.report import build_methane, to_markdown
from backend.state import (
    Hazard,
    Incident,
    IncidentStatus,
    IncidentType,
    UnitStatus,
    store,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crisis_command.main")


# ---------------------------------------------------------------------------
# Lifespan Hook
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Crisis Command state store with demo dataset...")
    store.load_demo()
    # Trigger initial solve for seeded incidents
    try:
        await replan({"kind": "manual", "payload": "initial_seed"})
    except Exception as exc:
        logger.warning("Initial plan computation deferred: %s", exc)
    yield
    logger.info("Crisis Command shutting down.")


app = FastAPI(
    title="Crisis Command Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class TextReportRequest(BaseModel):
    text: str = Field(min_length=1)


class ManualIncidentRequest(BaseModel):
    type: IncidentType
    severity: int = Field(ge=1, le=5)
    people_affected: int = Field(default=1, ge=0)
    lat: float
    lng: float
    location_text: str = Field(min_length=1)


class ConfirmIncidentRequest(BaseModel):
    type: IncidentType
    severity: int = Field(ge=1, le=5)
    people_affected: int = Field(default=1, ge=0)
    lat: float
    lng: float
    location_text: str = Field(min_length=1)
    notes: Optional[str] = ""
    source: Literal["text", "voice", "map", "seed"] = "text"


class UnitStatusRequest(BaseModel):
    status: UnitStatus


class HazardRequest(BaseModel):
    lat: float
    lng: float
    radius_m: float = Field(gt=0.0)
    kind: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    approve: bool


# ---------------------------------------------------------------------------
# Helper: Enrich Snapshot
# ---------------------------------------------------------------------------
def _get_enriched_state() -> Dict[str, Any]:
    snap = store.snapshot()
    units_list = list(store.units.values())
    gaps = coverage_gaps(units_list, settings.demo_bbox)
    staging = suggest_staging(units_list, gaps)
    snap["coverage"] = gaps
    snap["staging_suggestions"] = staging
    return snap


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    """Returns the full store snapshot enriched with coverage gaps and staging suggestions."""
    return _get_enriched_state()


@app.post("/api/incidents/text")
async def create_incident_from_text(payload: TextReportRequest):
    """Processes natural language text intake.

    If confidence is low or coordinates cannot be resolved, returns structured extraction
    for operator confirmation. Otherwise adds incident and triggers autonomous replanning.
    """
    extraction: IntakeResult = await extract(text=payload.text)

    if extraction.needs_confirmation or extraction.lat is None or extraction.lng is None or extraction.incident_type is None:
        return {
            "status": "needs_confirmation",
            "extraction": extraction.model_dump(),
        }

    inc = Incident(
        id=store.next_incident_id(),
        type=extraction.incident_type,
        severity=extraction.severity or 3,
        people_affected=extraction.people_affected or 1,
        lat=extraction.lat,
        lng=extraction.lng,
        location_text=extraction.location_text,
        confidence=extraction.confidence,
        needs_confirmation=False,
        source="text",
    )
    store.add_incident(inc)
    plan_result = await replan({"kind": "new_incident", "payload": inc})

    return {
        "status": "created",
        "incident": inc.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/incidents/audio")
async def create_incident_from_audio(file: UploadFile = File(...)):
    """Transcribes and extracts emergency dispatches from voice recordings."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    mime_type = file.content_type or "audio/webm"
    extraction: IntakeResult = await extract(audio_bytes=audio_bytes, mime_type=mime_type)

    if extraction.needs_confirmation or extraction.lat is None or extraction.lng is None or extraction.incident_type is None:
        return {
            "status": "needs_confirmation",
            "extraction": extraction.model_dump(),
        }

    inc = Incident(
        id=store.next_incident_id(),
        type=extraction.incident_type,
        severity=extraction.severity or 3,
        people_affected=extraction.people_affected or 1,
        lat=extraction.lat,
        lng=extraction.lng,
        location_text=extraction.location_text,
        confidence=extraction.confidence,
        needs_confirmation=False,
        source="voice",
    )
    store.add_incident(inc)
    plan_result = await replan({"kind": "new_incident", "payload": inc})

    return {
        "status": "created",
        "incident": inc.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/incidents/manual")
async def create_incident_manual(req: ManualIncidentRequest):
    """Direct incident insertion (e.g. clicking coordinates directly on the Leaflet map)."""
    inc = Incident(
        id=store.next_incident_id(),
        type=req.type,
        severity=req.severity,
        people_affected=req.people_affected,
        lat=req.lat,
        lng=req.lng,
        location_text=req.location_text,
        confidence=1.0,
        needs_confirmation=False,
        source="map",
    )
    store.add_incident(inc)
    plan_result = await replan({"kind": "new_incident", "payload": inc})

    return {
        "status": "created",
        "incident": inc.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/incidents/confirm")
async def confirm_incident(req: ConfirmIncidentRequest):
    """Commits an incident verified or corrected by human operator."""
    inc = Incident(
        id=store.next_incident_id(),
        type=req.type,
        severity=req.severity,
        people_affected=req.people_affected,
        lat=req.lat,
        lng=req.lng,
        location_text=req.location_text,
        confidence=1.0,
        needs_confirmation=False,
        source=req.source,
    )
    store.add_incident(inc)
    plan_result = await replan({"kind": "new_incident", "payload": inc})

    return {
        "status": "confirmed_and_created",
        "incident": inc.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/resources/{unit_id}/status")
async def update_resource_status(unit_id: str, req: UnitStatusRequest):
    """Updates unit availability or marks it down, triggering replanning."""
    unit = store.units.get(unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"Unit '{unit_id}' not found.")

    unit.status = req.status
    if req.status in (UnitStatus.down, UnitStatus.available):
        unit.assigned_incident_id = None

    trigger_kind = "unit_down" if req.status == UnitStatus.down else "manual"
    store.log_event("system", f"Unit {unit_id} ({unit.name}) status changed to {req.status.value}.")
    plan_result = await replan({"kind": trigger_kind, "payload": {"unit_id": unit_id}})

    return {
        "status": "updated",
        "unit": unit.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/units/{unit_id}/arrived")
async def mark_unit_arrived(unit_id: str):
    """Marks a responding unit as arrived on scene (busy)."""
    unit = store.units.get(unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"Unit '{unit_id}' not found.")

    inc_id = unit.assigned_incident_id
    unit.status = UnitStatus.busy
    store.log_event("system", f"Unit {unit.id} arrived on scene at incident {inc_id or 'unknown'}. Transitioned to BUSY.")

    return {
        "status": "arrived",
        "unit": unit.model_dump(),
    }


@app.post("/api/hazards")
async def create_hazard(req: HazardRequest):
    """Registers an active hazard/exclusion zone and updates travel corridors."""
    hazard = Hazard(
        id=store.next_hazard_id(),
        lat=req.lat,
        lng=req.lng,
        radius_m=req.radius_m,
        kind=req.kind,
    )
    store.add_hazard(hazard)
    plan_result = await replan({"kind": "hazard", "payload": hazard.model_dump()})

    return {
        "status": "hazard_created",
        "hazard": hazard.model_dump(),
        "plan_result": plan_result,
    }


@app.post("/api/approval/{thread_id}")
async def resolve_plan_approval(thread_id: str, req: ApprovalRequest):
    """Submits the operator decision (approve or reject) for an interrupted plan proposal."""
    try:
        result = await resolve_approval(thread_id=thread_id, approve=req.approve)
        return result
    except Exception as exc:
        logger.error("Failed to resolve approval for %s: %s", thread_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Approval resolution error: {str(exc)}"
        )


@app.post("/api/replan")
async def manual_replan():
    """Forces an immediate reassessment across all incidents and available units."""
    result = await replan({"kind": "manual", "payload": "operator_triggered"})
    return {"status": "replan_complete", "result": result}


@app.post("/api/scenario/reset")
async def reset_scenario():
    """Wipes in-memory state back to empty."""
    store.reset()
    return {"status": "reset", "state": _get_enriched_state()}


@app.post("/api/scenario/load-demo")
async def load_demo_scenario():
    """Reloads baseline resources and seeds initial incidents."""
    store.load_demo()
    plan_result = await replan({"kind": "manual", "payload": "load_demo"})
    return {
        "status": "demo_loaded",
        "state": _get_enriched_state(),
        "plan_result": plan_result,
    }


@app.get("/api/report/methane", response_class=PlainTextResponse)
async def get_methane_report():
    """Generates standardized METHANE Situation Report in downloadable Markdown format."""
    rep = build_methane(store)
    return to_markdown(rep)


@app.get("/api/stream")
async def event_stream(request: Request):
    """Server-Sent Events stream delivering store updates and keep-alive heartbeats."""
    queue = store.subscribe()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Emit full state snapshot upon establishing connection
            init_payload = json.dumps({"type": "init", "data": _get_enriched_state()})
            yield f"data: {init_payload}\n\n"

            while True:
                # Disconnection check
                if await request.is_disconnected():
                    break

                try:
                    # 15s timeout for keep-alive ping
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    msg = json.dumps(payload)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive SSE comment
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            store.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Static Files Mount for Frontend
# ---------------------------------------------------------------------------
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.warning("Frontend directory not located at %s", frontend_dir)