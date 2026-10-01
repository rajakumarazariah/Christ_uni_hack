"""backend/harm_curves.py
Expected harm response functions as a function of delay time (minutes).

DISCLAIMER & CLINICAL NOTE:
The mathematical curves and parameterizations defined here are illustrative, tunable
operational assumptions developed strictly for prototype decision modeling in Crisis
Command. They are NOT validated clinical prognosis datasets, triaging standards, or
medical guidelines. In production or real-world emergency dispatch, harm parameters
must be calibrated against vetted epidemiological data, local emergency service
protocols, and empirical casualty-survival statistics.
"""

from __future__ import annotations

import math
from typing import Any, Dict
from backend.state import IncidentType


# ---------------------------------------------------------------------------
# Tunable Model Parameters (No magic numbers inside equation bodies)
# ---------------------------------------------------------------------------

CURVES: Dict[IncidentType, Dict[str, Any]] = {
    # Out-of-hospital cardiac arrest: every minute without defibrillation/CPR degrades survival sharply.
    # Modeled via standard logistic curve: 1 / (1 + exp(-k * (t - t0)))
    IncidentType.cardiac_arrest: {
        "model": "logistic",
        "t0": 6.0,          # Midpoint inflection in minutes
        "k": 0.65,          # Steepness growth factor (near 1.0 by ~12 min)
        "description": "Steep logistic: high mortality risk beyond 6 min, near 1.0 at 12 min.",
    },
    # Golden hour trauma dynamics: gradual loss of clinical stability, saturating near 60 min.
    # Modeled as: min(1.0, slope * t)
    IncidentType.major_trauma: {
        "model": "linear_sat",
        "saturation_time": 60.0,
        "description": "Linear golden-hour decay: saturates to max harm at 60 min.",
    },
    # Structure/urban fires: flashover risks and exponential spread kinetics up to 25 min.
    # Modeled as: min(1.0, (exp(alpha * t) - 1) / (exp(alpha * t_sat) - 1))
    IncidentType.fire: {
        "model": "exponential",
        "saturation_time": 25.0,
        "alpha": 0.12,
        "description": "Exponential growth: fire spread accelerating, max harm at 25 min.",
    },
    # Entrapment in structural collapses: initial survival high during golden window (~30 min),
    # then stepped increase due to asphyxiation, crush syndrome, or structural shifts.
    IncidentType.building_collapse: {
        "model": "step_critical",
        "critical_window": 30.0,
        "baseline_harm": 0.15,
        "post_window_base": 0.70,
        "saturation_time": 60.0,
        "description": "Critical window: low harm up to 30 min, then sharp step increase.",
    },
    # Mass relocation / evacuation: logistical urgency escalates steadily over extended hours.
    IncidentType.evacuation: {
        "model": "linear_sat",
        "saturation_time": 120.0,
        "description": "Slow linear growth: logistical degradation over extended horizon.",
    },
    # Non-life-threatening lacerations, sprains, or superficial contusions.
    IncidentType.minor_injury: {
        "model": "capped_linear",
        "saturation_time": 60.0,
        "cap": 0.20,
        "description": "Very slow linear progression: capped at 0.20 maximum harm.",
    },
}


def harm(incident_type: IncidentType | str, minutes: float) -> float:
    """Computes expected harm in the closed interval [0.0, 1.0] for a given response time.

    Args:
        incident_type: The enum or string key for the incident type.
        minutes: Estimated response / travel time in minutes. Negative values treated as 0.0.

    Returns:
        Float value between 0.0 (no additional harm) and 1.0 (maximum expected harm/loss).
    """
    if isinstance(incident_type, str):
        try:
            incident_type = IncidentType(incident_type)
        except ValueError:
            # Fallback to major trauma behavior if unknown
            incident_type = IncidentType.major_trauma

    params = CURVES.get(incident_type, CURVES[IncidentType.major_trauma])
    t = max(0.0, float(minutes))
    model = params["model"]

    if model == "logistic":
        t0 = params["t0"]
        k = params["k"]
        # Standard logistic function f(t) = 1 / (1 + exp(-k * (t - t0)))
        val = 1.0 / (1.0 + math.exp(-k * (t - t0)))
        return float(min(1.0, max(0.0, val)))

    if model == "linear_sat":
        t_sat = params["saturation_time"]
        val = t / t_sat if t_sat > 0 else 1.0
        return float(min(1.0, max(0.0, val)))

    if model == "exponential":
        t_sat = params["saturation_time"]
        alpha = params["alpha"]
        if t >= t_sat:
            return 1.0
        # Scaled so harm(0) = 0.0 and harm(t_sat) = 1.0
        denom = math.exp(alpha * t_sat) - 1.0
        if denom <= 0:
            return 1.0
        val = (math.exp(alpha * t) - 1.0) / denom
        return float(min(1.0, max(0.0, val)))

    if model == "step_critical":
        t_crit = params["critical_window"]
        t_sat = params["saturation_time"]
        base = params["baseline_harm"]
        post = params["post_window_base"]

        if t <= t_crit:
            # Slow progression up to baseline_harm
            val = (t / t_crit) * base
        else:
            # Step jump up to post_window_base, then linear progression to 1.0
            excess_t = t - t_crit
            span = max(1.0, t_sat - t_crit)
            val = post + (excess_t / span) * (1.0 - post)
        return float(min(1.0, max(0.0, val)))

    if model == "capped_linear":
        t_sat = params["saturation_time"]
        cap = params["cap"]
        progress = min(1.0, t / t_sat if t_sat > 0 else 1.0)
        return float(min(cap, max(0.0, progress * cap)))

    return 0.5


def max_harm(incident_type: IncidentType | str, horizon: float = 60.0) -> float:
    """Calculates the harm cost for an incident left completely uncovered across a planning horizon."""
    return harm(incident_type, horizon)


def describe_curve(incident_type: IncidentType | str) -> str:
    """Returns a short human-readable string explaining the curve for UI tooltips."""
    if isinstance(incident_type, str):
        try:
            incident_type = IncidentType(incident_type)
        except ValueError:
            return "Standard golden-hour harm curve."
    params = CURVES.get(incident_type)
    if params and "description" in params:
        return params["description"]
    return "Standard response-delay curve."


if __name__ == "__main__":
    test_intervals = [2.0, 5.0, 10.0, 20.0, 40.0]

    header_cols = [f"{t:4.0f}m" for t in test_intervals]
    header = f"{'Incident Type':<20} | " + " | ".join(header_cols)
    divider = "-" * len(header)

    print("\nHARM RESPONSE PROFILES (Tunable Assumptions, [0.0 - 1.0])")
    print(divider)
    print(header)
    print(divider)

    for itype in IncidentType:
        row_vals = [f"{harm(itype, t):6.3f}" for t in test_intervals]
        print(f"{itype.value:<20} | " + " | ".join(row_vals))

    print(divider)
    print(f"{'Uncovered (60m max)':<20} | " + " | ".join([f"{max_harm(itype, 60.0):6.3f}" for itype in [IncidentType.cardiac_arrest, IncidentType.major_trauma, IncidentType.fire, IncidentType.building_collapse, IncidentType.evacuation, IncidentType.minor_injury]][:len(test_intervals)]))
    print()