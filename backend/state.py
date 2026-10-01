"""backend/state.py
Shared contract for Crisis Command: Pydantic v2 data models, enums,
and thread-safe in-memory Store singleton with SSE subscriber broadcasting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field, model_validator

from backend.config import data_path

logger = logging.getLogger("crisis_command.state")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IncidentType(str, Enum):
    cardiac_arrest = "cardiac_arrest"
    major_trauma = "major_trauma"
    minor_injury = "minor_injury"
    building_collapse = "building_collapse"
    fire = "fire"
    evacuation = "evacuation"


class UnitType(str, Enum):
    ambulance = "ambulance"
    rescue_team = "rescue_team"
    medical_unit = "medical_unit"


class UnitStatus(str, Enum):
    available = "available"
    en_route = "en_route"
    busy = "busy"
    down = "down"


class IncidentStatus(str, Enum):
    open = "open"
    assigned = "assigned"
    uncovered = "uncovered"
    resolved = "resolved"


# Default resource requirements by incident type
DEFAULT_UNITS_NEEDED: Dict[IncidentType, Dict[UnitType, int]] = {
    IncidentType.cardiac_arrest: {UnitType.ambulance: 1},
    IncidentType.major_trauma: {UnitType.ambulance: 1, UnitType.medical_unit: 1},
    IncidentType.minor_injury: {UnitType.medical_unit: 1},
    IncidentType.building_collapse: {UnitType.rescue_team: 2, UnitType.ambulance: 2},
    IncidentType.fire: {UnitType.rescue_team: 1, UnitType.ambulance: 1},
    IncidentType.evacuation: {UnitType.rescue_team: 2, UnitType.medical_unit: 1},
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Incident(BaseModel):
    id: str
    type: IncidentType
    severity: int = Field(ge=1, le=5)
    people_affected: int = Field(default=1, ge=0)
    lat: float
    lng: float
    location_text: str
    units_needed: Dict[UnitType, int] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_confirmation: bool = False
    source: Literal["text", "voice", "map", "seed"] = "text"
    status: IncidentStatus = IncidentStatus.open

    @model_validator(mode="before")
    @classmethod
    def populate_units_needed(cls, values: Any) -> Any:
        if isinstance(values, dict):
            u_needed = values.get("units_needed")
            inc_type = values.get("type")
            if not u_needed and inc_type:
                try:
                    typed_enum = IncidentType(inc_type)
                    values["units_needed"] = DEFAULT_UNITS_NEEDED.get(typed_enum, {UnitType.ambulance: 1}).copy()
                except ValueError:
                    values["units_needed"] = {UnitType.ambulance: 1}
        return values


class Unit(BaseModel):
    id: str
    type: UnitType
    name: str
    lat: float
    lng: float
    status: UnitStatus = UnitStatus.available
    assigned_incident_id: Optional[str] = None
    home_lat: float
    home_lng: float


class Facility(BaseModel):
    id: str
    kind: Literal["hospital", "shelter"]
    name: str
    lat: float
    lng: float
    capacity: int
    occupied: int = 0


class Hazard(BaseModel):
    id: str
    lat: float
    lng: float
    radius_m: float
    kind: str


class Assignment(BaseModel):
    unit_id: str
    incident_id: str
    eta_min: float
    harm: float


class Uncovered(BaseModel):
    incident_id: str
    unit_type: UnitType
    missing: int


class Plan(BaseModel):
    id: str
    version: int
    assignments: List[Assignment] = Field(default_factory=list)
    uncovered: List[Uncovered] = Field(default_factory=list)
    total_harm: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trigger: str = ""
    status: Literal["proposed", "approved", "rejected"] = "proposed"


class DiffChange(BaseModel):
    unit_id: str
    from_incident_id: Optional[str] = None
    to_incident_id: str
    eta_before: Optional[float] = None
    eta_after: float
    harm_delta: float


class PlanDiff(BaseModel):
    added: List[Assignment] = Field(default_factory=list)
    removed: List[Assignment] = Field(default_factory=list)
    changed: List[DiffChange] = Field(default_factory=list)
    unchanged_count: int = 0


class ReasoningCard(BaseModel):
    unit_id: str
    from_incident_id: Optional[str] = None
    to_incident_id: str
    text: str
    numbers: Dict[str, Any] = Field(default_factory=dict)
    needs_approval: bool = False


class Event(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: Literal["intake", "logistics", "solver", "command", "human", "system"]
    message: str


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

class Store:
    """Thread-safe, reactive in-memory state store for Crisis Command."""

    def __init__(self) -> None:
        self.incidents: Dict[str, Incident] = {}
        self.units: Dict[str, Unit] = {}
        self.facilities: Dict[str, Facility] = {}
        self.hazards: Dict[str, Hazard] = {}

        self.current_plan: Optional[Plan] = None
        self.previous_plan: Optional[Plan] = None
        self.pending_plan: Optional[Plan] = None

        self.events: List[Event] = []
        self._subscribers: Set[asyncio.Queue] = set()

        self._incident_counter: int = 0
        self._hazard_counter: int = 0
        self._plan_counter: int = 0

    def next_incident_id(self) -> str:
        self._incident_counter += 1
        return f"INC-{self._incident_counter}"

    def next_hazard_id(self) -> str:
        self._hazard_counter += 1
        return f"HAZ-{self._hazard_counter}"

    def next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"PLAN-{self._plan_counter}"

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, payload: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue is full; dropping event.")

    def log_event(
        self,
        agent: Literal["intake", "logistics", "solver", "command", "human", "system"],
        message: str
    ) -> Event:
        event = Event(agent=agent, message=message)
        self.events.append(event)
        self._broadcast({"type": "event", "data": event.model_dump()})
        return event

    def add_incident(self, incident: Incident) -> None:
        self.incidents[incident.id] = incident
        self.log_event("intake", f"New incident reported: {incident.id} ({incident.type.value}, severity {incident.severity})")
        self._broadcast({"type": "incident_added", "data": incident.model_dump()})

    def set_unit_status(
        self,
        unit_id: str,
        status: UnitStatus,
        assigned_incident_id: Optional[str] = None
    ) -> Optional[Unit]:
        unit = self.units.get(unit_id)
        if not unit:
            return None
        unit.status = status
        unit.assigned_incident_id = assigned_incident_id
        self._broadcast({"type": "unit_updated", "data": unit.model_dump()})
        return unit

    def add_hazard(self, hazard: Hazard) -> None:
        self.hazards[hazard.id] = hazard
        self.log_event("system", f"Hazard active: {hazard.id} ({hazard.kind}, radius {hazard.radius_m}m)")
        self._broadcast({"type": "hazard_added", "data": hazard.model_dump()})

    def commit_plan(self, plan: Plan) -> None:
        plan.status = "approved"
        self.previous_plan = self.current_plan
        self.current_plan = plan
        self.pending_plan = None

        # Apply assignments to units
        assigned_units = {a.unit_id: a.incident_id for a in plan.assignments}
        for u_id, unit in self.units.items():
            if u_id in assigned_units:
                unit.status = UnitStatus.en_route
                unit.assigned_incident_id = assigned_units[u_id]
            elif unit.status == UnitStatus.en_route:
                unit.status = UnitStatus.available
                unit.assigned_incident_id = None

        # Update incident statuses
        assigned_incidents = {a.incident_id for a in plan.assignments}
        uncovered_incidents = {u.incident_id for u in plan.uncovered}

        for inc_id, inc in self.incidents.items():
            if inc.status == IncidentStatus.resolved:
                continue
            if inc_id in assigned_incidents:
                inc.status = IncidentStatus.assigned
            elif inc_id in uncovered_incidents:
                inc.status = IncidentStatus.uncovered
            else:
                inc.status = IncidentStatus.open

        self.log_event("command", f"Plan {plan.id} (v{plan.version}) committed with {len(plan.assignments)} assignments.")
        self._broadcast({"type": "plan_committed", "data": plan.model_dump()})

    def snapshot(self) -> dict:
        """Full state dictionary for frontend consumption and map rendering."""
        return {
            "incidents": [inc.model_dump() for inc in sorted(self.incidents.values(), key=lambda x: x.id)],
            "units": [u.model_dump() for u in sorted(self.units.values(), key=lambda x: x.id)],
            "facilities": [f.model_dump() for f in sorted(self.facilities.values(), key=lambda x: x.id)],
            "hazards": [h.model_dump() for h in sorted(self.hazards.values(), key=lambda x: x.id)],
            "current_plan": self.current_plan.model_dump() if self.current_plan else None,
            "previous_plan": self.previous_plan.model_dump() if self.previous_plan else None,
            "pending_plan": self.pending_plan.model_dump() if self.pending_plan else None,
            "events": [e.model_dump() for e in self.events[-50:]],
        }

    def reset(self) -> None:
        self.incidents.clear()
        self.units.clear()
        self.facilities.clear()
        self.hazards.clear()
        self.current_plan = None
        self.previous_plan = None
        self.pending_plan = None
        self.events.clear()
        self._incident_counter = 0
        self._hazard_counter = 0
        self._plan_counter = 0
        self.log_event("system", "State store reset to blank.")

    def load_demo(self) -> None:
        self.reset()

        # Load resources.json (units and facilities)
        res_file = data_path("resources.json")
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                for u in res_data.get("units", []):
                    unit_obj = Unit(**u)
                    self.units[unit_obj.id] = unit_obj
                for fac in res_data.get("facilities", []):
                    fac_obj = Facility(**fac)
                    self.facilities[fac_obj.id] = fac_obj
        else:
            logger.warning("Resources file not found: %s", res_file)

        # Load incidents_seed.json
        inc_file = data_path("incidents_seed.json")
        if inc_file.exists():
            with open(inc_file, "r", encoding="utf-8") as f:
                inc_data = json.load(f)
                for inc in inc_data:
                    inc_obj = Incident(**inc)
                    self.incidents[inc_obj.id] = inc_obj
                    # Track highest counter to avoid ID collision
                    if inc_obj.id.startswith("INC-"):
                        try:
                            num = int(inc_obj.id.split("-")[1])
                            self._incident_counter = max(self._incident_counter, num)
                        except ValueError:
                            pass
        else:
            logger.warning("Incidents seed file not found: %s", inc_file)

        self.log_event("system", f"Demo scenario loaded: {len(self.units)} units, {len(self.incidents)} incidents.")
        self._broadcast({"type": "reset", "data": self.snapshot()})


# Shared global store instance
store = Store()