"""backend/logistics.py
Logistics engine for Crisis Command:
- ETA matrix generation via OSRM table service with Haversine fallback and hazard penalties.
- Spatial grid coverage gap analysis.
- Proactive unit staging/redeployment recommendations.
- Nearest destination facility routing (hospitals and shelters) with capacity tracking.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import httpx
import numpy as np

from backend.config import settings
from backend.state import Facility, Hazard, Incident, Unit, UnitStatus

# In-memory ETA cache keyed by coordinate signature: (src_lat, src_lng, dst_lat, dst_lng)
_ETA_CACHE: Dict[Tuple[float, float, float, float], float] = {}


# ---------------------------------------------------------------------------
# Geometric & Routing Utilities
# ---------------------------------------------------------------------------

def haversine_distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Computes the great-circle distance between two points on Earth in kilometers."""
    r = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def haversine_eta_min(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
    detour_factor: float = 1.4,
    speed_kmph: Optional[float] = None,
) -> float:
    """Computes road travel time in minutes using Haversine distance and an urban detour factor."""
    speed = speed_kmph or settings.avg_speed_kmph
    dist_km = haversine_distance_km(lat1, lng1, lat2, lng2) * detour_factor
    return max(0.5, (dist_km / max(1.0, speed)) * 60.0)


def distance_point_to_segment_m(
    p_lat: float, p_lng: float,
    a_lat: float, a_lng: float,
    b_lat: float, b_lng: float
) -> float:
    """Computes minimum distance in meters from point P to line segment AB using flat-earth projection."""
    # Convert lat/lng differences to local metric coordinates (x: East, y: North)
    mean_lat_rad = math.radians((a_lat + b_lat + p_lat) / 3.0)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(mean_lat_rad)

    px = (p_lng - a_lng) * meters_per_deg_lng
    py = (p_lat - a_lat) * meters_per_deg_lat
    bx = (b_lng - a_lng) * meters_per_deg_lng
    by = (b_lat - a_lat) * meters_per_deg_lat

    seg_len_sq = bx * bx + by * by
    if seg_len_sq < 1e-6:
        return math.sqrt(px * px + py * py)

    # Project P onto AB, clamped to the unit interval [0, 1]
    t = max(0.0, min(1.0, (px * bx + py * by) / seg_len_sq))
    proj_x = t * bx
    proj_y = t * by

    dx = px - proj_x
    dy = py - proj_y
    return math.sqrt(dx * dx + dy * dy)


def segment_intersects_hazard(
    unit_lat: float, unit_lng: float,
    inc_lat: float, inc_lng: float,
    hazard: Hazard
) -> bool:
    """Checks whether the straight-line response vector passes within the hazard radius."""
    dist_m = distance_point_to_segment_m(
        hazard.lat, hazard.lng,
        unit_lat, unit_lng,
        inc_lat, inc_lng
    )
    return dist_m <= hazard.radius_m


# ---------------------------------------------------------------------------
# 1. ETA Matrix Builder
# ---------------------------------------------------------------------------

async def build_eta_matrix(
    units: Sequence[Unit],
    incidents: Sequence[Incident],
    hazards: Sequence[Hazard] = ()
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    """Builds a [num_units x num_incidents] matrix of travel times in minutes.

    Attempts OSRM /table/v1/driving in a single batch call. Falls back to Haversine
    on timeout, network failure, or if settings.use_osrm is False. Applies detour
    penalties if route intersects any active hazard zone.
    """
    n_units = len(units)
    n_inc = len(incidents)

    if n_units == 0 or n_inc == 0:
        return np.zeros((n_units, n_inc), dtype=float), []

    matrix = np.zeros((n_units, n_inc), dtype=float)
    meta: List[List[Dict[str, Any]]] = [
        [{} for _ in range(n_inc)] for _ in range(n_units)
    ]

    # Pre-populate with fallback calculation first
    for i, u in enumerate(units):
        for j, inc in enumerate(incidents):
            eta_fb = haversine_eta_min(u.lat, u.lng, inc.lat, inc.lng)
            matrix[i, j] = eta_fb
            meta[i][j] = {
                "source": "fallback",
                "hazard_penalty": False,
                "hazard_ids": []
            }

    # Attempt OSRM Table Service
    if settings.use_osrm:
        # Check cache first for all pairs
        all_cached = True
        for i, u in enumerate(units):
            for j, inc in enumerate(incidents):
                cache_key = (
                    round(u.lat, 4), round(u.lng, 4),
                    round(inc.lat, 4), round(inc.lng, 4)
                )
                if cache_key in _ETA_CACHE:
                    matrix[i, j] = _ETA_CACHE[cache_key]
                    meta[i][j]["source"] = "osrm_cache"
                else:
                    all_cached = False

        if not all_cached:
            # Build OSRM request coordinate string: {lng,lat};...
            # Format: all units (sources: 0..n_units-1) followed by incidents (destinations: n_units..n_units+n_inc-1)
            coords: List[str] = [f"{u.lng:.6f},{u.lat:.6f}" for u in units]
            coords.extend([f"{inc.lng:.6f},{inc.lat:.6f}" for inc in incidents])
            coords_str = ";".join(coords)

            src_indices = ";".join(str(i) for i in range(n_units))
            dst_indices = ";".join(str(n_units + j) for j in range(n_inc))

            url = (
                f"{settings.osrm_base_url}/table/v1/driving/{coords_str}"
                f"?sources={src_indices}&destinations={dst_indices}"
            )

            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        durations = data.get("durations")
                        if durations and len(durations) == n_units:
                            for i, u in enumerate(units):
                                for j, inc in enumerate(incidents):
                                    d_sec = durations[i][j]
                                    if d_sec is not None:
                                        eta_min = max(0.5, float(d_sec) / 60.0)
                                        matrix[i, j] = eta_min
                                        meta[i][j]["source"] = "osrm"

                                        cache_key = (
                                            round(u.lat, 4), round(u.lng, 4),
                                            round(inc.lat, 4), round(inc.lng, 4)
                                        )
                                        _ETA_CACHE[cache_key] = eta_min
            except Exception:
                # Retains fallback Haversine values already in matrix
                pass

    # Hazard intersection check and penalty: ETA * 1.5 + 3.0 min
    if hazards:
        for i, u in enumerate(units):
            for j, inc in enumerate(incidents):
                intersected = [
                    h.id for h in hazards
                    if segment_intersects_hazard(u.lat, u.lng, inc.lat, inc.lng, h)
                ]
                if intersected:
                    matrix[i, j] = (matrix[i, j] * 1.5) + 3.0
                    meta[i][j]["hazard_penalty"] = True
                    meta[i][j]["hazard_ids"] = intersected

    return matrix, meta


# ---------------------------------------------------------------------------
# 2. Coverage Gaps Analysis
# ---------------------------------------------------------------------------

def coverage_gaps(
    units: Sequence[Unit],
    bbox: Tuple[float, float, float, float],
    target_min: Optional[int] = None,
    grid: int = 15
) -> Dict[str, Any]:
    """Overlays a grid on bbox, computing nearest ETA from available, unassigned units."""
    south, west, north, east = bbox
    threshold = target_min or settings.target_response_min

    # Filter to available units with no current incident assignment
    avail_units = [
        u for u in units
        if u.status == UnitStatus.available and u.assigned_incident_id is None
    ]

    lat_steps = np.linspace(south, north, grid)
    lng_steps = np.linspace(west, east, grid)

    cells: List[Dict[str, Any]] = []
    uncovered_count = 0
    total_cells = grid * grid

    for lat in lat_steps:
        for lng in lng_steps:
            c_lat = float(lat)
            c_lng = float(lng)

            if not avail_units:
                cells.append({
                    "lat": c_lat,
                    "lng": c_lng,
                    "covered": False,
                    "nearest_eta": 999.0
                })
                uncovered_count += 1
                continue

            nearest_eta = min(
                haversine_eta_min(u.lat, u.lng, c_lat, c_lng)
                for u in avail_units
            )
            is_covered = nearest_eta <= threshold
            if not is_covered:
                uncovered_count += 1

            cells.append({
                "lat": c_lat,
                "lng": c_lng,
                "covered": is_covered,
                "nearest_eta": round(nearest_eta, 1)
            })

    uncovered_share = round(uncovered_count / max(1, total_cells), 4)

    return {
        "cells": cells,
        "uncovered_share": uncovered_share,
        "uncovered_count": uncovered_count,
        "total_cells": total_cells,
        "threshold_min": threshold
    }


# ---------------------------------------------------------------------------
# 3. Proactive Staging Recommendations
# ---------------------------------------------------------------------------

def suggest_staging(
    units: Sequence[Unit],
    gaps: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Suggests redeploying an idle unit from an over-covered area to the centroid of an uncovered gap cluster."""
    if gaps.get("uncovered_share", 0.0) <= 0.25:
        return []

    uncovered_cells = [c for c in gaps.get("cells", []) if not c["covered"]]
    if not uncovered_cells:
        return []

    avail_units = [
        u for u in units
        if u.status == UnitStatus.available and u.assigned_incident_id is None
    ]
    if len(avail_units) < 2:
        return []

    # Calculate centroid of the uncovered cluster
    target_lat = sum(c["lat"] for c in uncovered_cells) / len(uncovered_cells)
    target_lng = sum(c["lng"] for c in uncovered_cells) / len(uncovered_cells)

    # Find the idle unit with highest local density of neighboring units (most redundant)
    redundancy_scores: List[Tuple[float, Unit]] = []
    for u in avail_units:
        # Redundancy = sum of inverse distances to other available units
        score = sum(
            1.0 / max(0.5, haversine_distance_km(u.lat, u.lng, other.lat, other.lng))
            for other in avail_units if other.id != u.id
        )
        redundancy_scores.append((score, u))

    redundancy_scores.sort(key=lambda x: (x[0], x[1].id), reverse=True)
    best_candidate = redundancy_scores[0][1]

    return [{
        "unit_id": best_candidate.id,
        "unit_name": best_candidate.name,
        "current_lat": best_candidate.lat,
        "current_lng": best_candidate.lng,
        "target_lat": round(target_lat, 4),
        "target_lng": round(target_lng, 4),
        "reason": (
            f"Coverage deficit is {int(gaps['uncovered_share'] * 100)}%. "
            f"Move {best_candidate.name} from dense cluster to cover sector centroid."
        )
    }]


# ---------------------------------------------------------------------------
# 4. Destination Facility Routing
# ---------------------------------------------------------------------------

def pick_destination(
    incident: Incident,
    facilities: Sequence[Facility],
    kind: Literal["hospital", "shelter"]
) -> Optional[Facility]:
    """Finds the nearest facility of the requested kind with available capacity."""
    candidates = [
        f for f in facilities
        if f.kind == kind and (f.capacity - f.occupied) > 0
    ]
    if not candidates:
        return None

    # Deterministic sorting: distance first, tie-break by facility id
    candidates.sort(
        key=lambda f: (
            haversine_distance_km(incident.lat, incident.lng, f.lat, f.lng),
            f.id
        )
    )
    return candidates[0]

async def fetch_street_route(lat1: float, lng1: float, lat2: float, lng2: float) -> list[list[float]]:
    """Fetches real road waypoints [[lat, lng], ...] from OSRM, falling back to a straight line."""
    if not settings.use_osrm:
        return [[lat1, lng1], [lat2, lng2]]

    url = f"{settings.osrm_base_url}/route/v1/driving/{lng1:.6f},{lat1:.6f};{lng2:.6f},{lat2:.6f}?overview=full&geometries=geojson"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if routes:
                    # GeoJSON is [lng, lat], convert to Leaflet's expected [lat, lng]
                    coords = routes[0]["geometry"]["coordinates"]
                    return [[pt[1], pt[0]] for pt in coords]
    except Exception:
        pass

    return [[lat1, lng1], [lat2, lng2]]


# ---------------------------------------------------------------------------
# Standalone Verification Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _demo() -> None:
        test_units = [
            Unit(
                id="AMB-1", type="ambulance", name="Cantonment 1",
                lat=10.8035, lng=78.6885, home_lat=10.8035, home_lng=78.6885
            ),
            Unit(
                id="AMB-2", type="ambulance", name="Thillai Nagar 2",
                lat=10.8290, lng=78.6830, home_lat=10.8290, home_lng=78.6830
            ),
            Unit(
                id="AMB-3", type="ambulance", name="Srirangam 3",
                lat=10.8620, lng=78.6920, home_lat=10.8620, home_lng=78.6920
            ),
        ]
        test_incidents = [
            Incident(
                id="INC-1", type="cardiac_arrest", severity=5,
                lat=10.8015, lng=78.6820, location_text="Cantonment Circle"
            ),
            Incident(
                id="INC-2", type="major_trauma", severity=3,
                lat=10.8605, lng=78.6908, location_text="Srirangam Gate"
            ),
        ]
        test_hazards = [
            Hazard(id="HAZ-1", lat=10.8150, lng=78.6850, radius_m=500.0, kind="flooding")
        ]

        print("--- Building ETA Matrix ---")
        matrix, meta = await build_eta_matrix(test_units, test_incidents, test_hazards)
        for i, u in enumerate(test_units):
            row = [f"{matrix[i, j]:5.1f}m ({meta[i][j]['source']})" for j in range(len(test_incidents))]
            print(f"{u.id:<6} -> " + " | ".join(row))

        print("\n--- Coverage Gap Analysis ---")
        gaps = coverage_gaps(test_units, settings.demo_bbox, grid=5)
        print(f"Uncovered share: {gaps['uncovered_share'] * 100:.1f}%")

        print("\n--- Staging Suggestions ---")
        staging = suggest_staging(test_units, gaps)
        print(staging if staging else "Coverage adequate; no staging moves required.")

    asyncio.run(_demo())