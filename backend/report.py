"""backend/report.py
METHANE standardized Situation Report (SitRep) generator for Crisis Command.
Builds structured operational intelligence reports and formats clean, downloadable
Markdown summaries complete with casualty estimations and a timestamped audit trail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.state import IncidentStatus, Store, UnitStatus, UnitType


def build_methane(store_instance: Store) -> Dict[str, Any]:
    """Assembles a structured METHANE situation report from the current state store.

    METHANE protocol sections:
    - M: Major Incident Standby / Declared
    - E: Exact Locations
    - T: Types of Incident
    - H: Hazards Present or Suspected
    - A: Access Routes and Restrictions
    - N: Number and Severity of Casualties (Estimated)
    - E: Emergency Services Present and Required
    """
    active_incidents = [
        inc for inc in store_instance.incidents.values()
        if inc.status != IncidentStatus.resolved
    ]
    all_units = list(store_instance.units.values())
    hazards = list(store_instance.hazards.values())
    current_plan = store_instance.current_plan

    # M - Major Incident Declaration
    has_high_severity = any(inc.severity >= 4 for inc in active_incidents)
    has_multiple_open = len(active_incidents) >= 3
    is_major_incident = has_high_severity or has_multiple_open
    declaration_reason = []
    if has_high_severity:
        declaration_reason.append("presence of critical severity (>=4) incidents")
    if has_multiple_open:
        declaration_reason.append(f"{len(active_incidents)} concurrent active incidents")
    reason_str = ", ".join(declaration_reason) if declaration_reason else "routine operational threshold"

    # E - Exact Locations
    locations = [
        {
            "id": inc.id,
            "location_text": inc.location_text,
            "coordinates": f"{inc.lat:.4f}, {inc.lng:.4f}",
            "status": inc.status.value,
        }
        for inc in sorted(active_incidents, key=lambda x: x.id)
    ]

    # T - Types of Incident
    types_breakdown: Dict[str, int] = {}
    for inc in active_incidents:
        key = inc.type.value.replace("_", " ").title()
        types_breakdown[key] = types_breakdown.get(key, 0) + 1

    # H - Hazards
    hazard_list = [
        {
            "id": h.id,
            "kind": h.kind,
            "radius_m": h.radius_m,
            "coordinates": f"{h.lat:.4f}, {h.lng:.4f}",
        }
        for h in sorted(hazards, key=lambda x: x.id)
    ]

    # A - Access Routes and Restrictions
    access_warnings: List[str] = []
    if hazards:
        for h in hazards:
            access_warnings.append(
                f"Zone advisory around {h.id} ({h.kind}): Avoid routes within {h.radius_m:.0f}m of ({h.lat:.4f}, {h.lng:.4f}). Routing engine applies detour penalties."
            )
    else:
        access_warnings.append("All primary arterial and feeder corridors reported clear. No hazard exclusions active.")

    # N - Number of Casualties (Estimated)
    total_casualties = sum(inc.people_affected for inc in active_incidents)
    severity_breakdown: Dict[int, int] = {s: 0 for s in range(1, 6)}
    for inc in active_incidents:
        severity_breakdown[inc.severity] = severity_breakdown.get(inc.severity, 0) + inc.people_affected

    # E - Emergency Services Present and Required
    present_counts: Dict[str, int] = {ut.value: 0 for ut in UnitType}
    deployed_counts: Dict[str, int] = {ut.value: 0 for ut in UnitType}
    for u in all_units:
        if u.status == UnitStatus.available:
            present_counts[u.type.value] += 1
        elif u.status == UnitStatus.en_route:
            deployed_counts[u.type.value] += 1

    uncovered_demand: Dict[str, int] = {ut.value: 0 for ut in UnitType}
    if current_plan:
        for unc in current_plan.uncovered:
            uncovered_demand[unc.unit_type.value] += unc.missing
    else:
        # Tally directly from active incident needs if no plan is yet committed
        for inc in active_incidents:
            for ut, count in inc.units_needed.items():
                uncovered_demand[ut.value] += count

    # Audit Trail: Events timeline
    timeline = [
        {
            "timestamp": e.ts,
            "agent": e.agent,
            "message": e.message,
        }
        for e in store_instance.events
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "major_incident_declared": is_major_incident,
        "declaration_reason": reason_str,
        "locations": locations,
        "incident_types": types_breakdown,
        "hazards": hazard_list,
        "access": access_warnings,
        "casualties": {
            "total_estimated": total_casualties,
            "by_severity": severity_breakdown,
        },
        "services": {
            "available": present_counts,
            "en_route": deployed_counts,
            "uncovered_deficit": uncovered_demand,
        },
        "timeline": timeline,
    }


def to_markdown(report: Dict[str, Any]) -> str:
    """Formats the METHANE report dictionary into clean, downloadable Markdown."""
    lines: List[str] = []

    status_tag = "MAJOR INCIDENT DECLARED" if report["major_incident_declared"] else "ROUTINE STANDBY / MONITORING"
    lines.append(f"# Crisis Command Situation Report (METHANE)")
    lines.append(f"**Generated:** {report['generated_at']}  ")
    lines.append(f"**Status:** `{status_tag}`\n")
    lines.append("---")

    # M
    lines.append(f"### [M] Major Incident")
    if report["major_incident_declared"]:
        lines.append(f"**DECLARED** - Triggered by: {report['declaration_reason']}.\n")
    else:
        lines.append("STANDBY - Current operational load within baseline capacity.\n")

    # E
    lines.append("### [E] Exact Locations")
    if report["locations"]:
        lines.append("| Incident ID | Location / Landmark | Coordinates | Status |")
        lines.append("|---|---|---|---|")
        for loc in report["locations"]:
            lines.append(f"| **{loc['id']}** | {loc['location_text']} | `{loc['coordinates']}` | {loc['status']} |")
    else:
        lines.append("_No open incidents registered._")
    lines.append("")

    # T
    lines.append("### [T] Types of Incident")
    if report["incident_types"]:
        for inc_t, cnt in report["incident_types"].items():
            lines.append(f"- **{inc_t}**: {cnt} active scene(s)")
    else:
        lines.append("- None reported.")
    lines.append("")

    # H
    lines.append("### [H] Hazards Present or Suspected")
    if report["hazards"]:
        for h in report["hazards"]:
            lines.append(f"- **{h['id']}** (`{h['kind']}`): Perimeter {h['radius_m']:.0f}m at `{h['coordinates']}`")
    else:
        lines.append("- No environmental, chemical, or structural exclusion zones active.")
    lines.append("")

    # A
    lines.append("### [A] Access Routes & Restrictions")
    for acc in report["access"]:
        lines.append(f"- {acc}")
    lines.append("")

    # N
    lines.append("### [N] Number of Casualties (Estimated)")
    cas = report["casualties"]
    lines.append(f"- **Total Casualties (Estimated):** **{cas['total_estimated']}**")
    lines.append("  - Severity 5 (Life-Threatening): " + str(cas["by_severity"].get(5, 0)))
    lines.append("  - Severity 4 (Severe/Multiple): " + str(cas["by_severity"].get(4, 0)))
    lines.append("  - Severity 3 (Moderate/Evac): " + str(cas["by_severity"].get(3, 0)))
    lines.append("  - Severity 1-2 (Minor/Superficial): " + str(cas["by_severity"].get(1, 0) + cas["by_severity"].get(2, 0)))
    lines.append("")

    # E
    lines.append("### [E] Emergency Services Summary")
    srv = report["services"]
    lines.append("| Service Type | Available on Standby | En Route / On Scene | Uncovered Deficit |")
    lines.append("|---|:---:|:---:|:---:|")
    all_types = sorted(set(srv["available"].keys()) | set(srv["en_route"].keys()) | set(srv["uncovered_deficit"].keys()))
    for t in all_types:
        disp_name = t.replace("_", " ").title()
        avail = srv["available"].get(t, 0)
        deployed = srv["en_route"].get(t, 0)
        deficit = srv["uncovered_deficit"].get(t, 0)
        deficit_str = f"**+{deficit} needed**" if deficit > 0 else "0"
        lines.append(f"| **{disp_name}** | {avail} | {deployed} | {deficit_str} |")
    lines.append("")

    # Decision Timeline
    lines.append("---")
    lines.append("### Incident Audit & Decision Timeline")
    if report["timeline"]:
        lines.append("| UTC Time | Agent | Operational Log Entry |")
        lines.append("|---|---|---|")
        for ev in report["timeline"]:
            t_short = ev["timestamp"].split("T")[1][:8] if "T" in ev["timestamp"] else ev["timestamp"]
            lines.append(f"| `{t_short}` | **{ev['agent']}** | {ev['message']} |")
    else:
        lines.append("_No logged operational events._")
    lines.append("")

    return "\n".join(lines)