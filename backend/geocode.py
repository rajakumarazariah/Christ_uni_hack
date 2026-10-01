"""backend/geocode.py
Geocoding module for Crisis Command.
Resolves natural-language location descriptions by querying local landmark aliases
with fuzzy matching (difflib) first, falling back to OpenStreetMap Nominatim with
rate-limiting, strict bounding box constraints, timeouts, and in-memory caching.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.config import data_path, settings

logger = logging.getLogger("crisis_command.geocode")

# Global lock and timestamp to respect Nominatim's strict rate limit (>= 1.1s between calls)
_NOMINATIM_LOCK = asyncio.Lock()
_LAST_NOMINATIM_CALL: float = 0.0

# In-memory cache for Nominatim results: normalized_query -> result dict
_GEOCODE_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

# In-memory landmark database loaded on startup
_LANDMARKS: List[Dict[str, Any]] = []


def _normalize(text: str) -> str:
    """Lowercases and strips punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_landmarks() -> List[Dict[str, Any]]:
    """Loads landmark entries from backend/data/landmarks.json."""
    global _LANDMARKS
    if _LANDMARKS:
        return _LANDMARKS

    path = data_path("landmarks.json")
    if not path.exists():
        logger.warning("Landmarks file not found at %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            _LANDMARKS = json.load(f)
    except Exception as exc:
        logger.error("Failed to load landmarks.json: %s", exc)
        _LANDMARKS = []

    return _LANDMARKS


def _match_landmark(normalized_query: str, cutoff: float = 0.75) -> Optional[Dict[str, Any]]:
    """Performs full and fuzzy substring/token matching against landmark names and aliases."""
    landmarks = _load_landmarks()
    if not landmarks or not normalized_query:
        return None

    best_match: Optional[Dict[str, Any]] = None
    highest_ratio: float = 0.0
    matched_label: str = ""

    for item in landmarks:
        candidates = [item.get("name", "")] + item.get("aliases", [])
        for cand in candidates:
            cand_norm = _normalize(cand)
            if not cand_norm:
                continue

            # Exact or direct containment match
            if cand_norm == normalized_query or cand_norm in normalized_query or normalized_query in cand_norm:
                return {
                    "lat": float(item["lat"]),
                    "lng": float(item["lng"]),
                    "source": "landmark",
                    "confidence": 0.9,
                    "matched_name": item["name"],
                }

            # difflib fuzzy sequence comparison
            ratio = difflib.SequenceMatcher(None, normalized_query, cand_norm).ratio()
            if ratio > highest_ratio and ratio >= cutoff:
                highest_ratio = ratio
                best_match = item
                matched_label = item["name"]

    if best_match and highest_ratio >= cutoff:
        return {
            "lat": float(best_match["lat"]),
            "lng": float(best_match["lng"]),
            "source": "landmark",
            "confidence": 0.9,
            "matched_name": matched_label,
        }

    return None


async def resolve(location_text: str) -> Optional[Dict[str, Any]]:
    """Resolves an operator location string to geographic coordinates.

    Tries landmarks.json first via fuzzy matching. If unresolved, falls back to
    Nominatim bounded by the city bbox, rate-limited to 1 request per 1.1 seconds.
    Never invents coordinates; returns None if not found.
    """
    global _LAST_NOMINATIM_CALL

    if not location_text or not location_text.strip():
        return None

    norm_query = _normalize(location_text)

    # 1. Match against local landmark database
    landmark_match = _match_landmark(norm_query)
    if landmark_match:
        return landmark_match

    # Check cache for prior Nominatim lookups
    if norm_query in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[norm_query]

    # 2. Query Nominatim with rate limiting & bounding box
    south, west, north, east = settings.demo_bbox
    # Nominatim viewbox parameter: left,top,right,bottom (west, north, east, south)
    viewbox_str = f"{west:.6f},{north:.6f},{east:.6f},{south:.6f}"
    full_query = f"{location_text}, {settings.demo_city_name}"

    async with _NOMINATIM_LOCK:
        elapsed = time.monotonic() - _LAST_NOMINATIM_CALL
        if elapsed < 1.1:
            await asyncio.sleep(1.1 - elapsed)

        params = {
            "q": full_query,
            "format": "jsonv2",
            "limit": "1",
            "viewbox": viewbox_str,
            "bounded": "1",
        }
        headers = {
            "User-Agent": settings.nominatim_user_agent,
        }

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(
                    f"{settings.nominatim_url}/search",
                    params=params,
                    headers=headers,
                )
                _LAST_NOMINATIM_CALL = time.monotonic()

                if resp.status_code == 200:
                    results = resp.json()
                    if results and isinstance(results, list):
                        top = results[0]
                        resolved = {
                            "lat": float(top["lat"]),
                            "lng": float(top["lon"]),
                            "source": "nominatim",
                            "confidence": 0.6,
                            "matched_name": top.get("display_name", location_text),
                        }
                        _GEOCODE_CACHE[norm_query] = resolved
                        return resolved
        except Exception as exc:
            logger.warning("Nominatim geocoding failed for '%s': %s", full_query, exc)
            _LAST_NOMINATIM_CALL = time.monotonic()

    # 3. Nothing found or request timed out
    _GEOCODE_CACHE[norm_query] = None
    return None