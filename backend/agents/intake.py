"""backend/agents/intake.py
Incident extraction agent for Crisis Command using the google-genai SDK.
Handles unstructured voice (audio bytes) and text reports, strictly enforcing
structured JSON schema outputs, confidence scoring, landmark/Nominatim geocoding,
and offline keyword-based fallback when LLM_MOCK is active.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from backend.config import settings
from backend.geocode import resolve as geocode_resolve
from backend.state import IncidentType, store

logger = logging.getLogger("crisis_command.intake")

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class RawIntakeSchema(BaseModel):
    """Raw structured extraction target from Gemini."""
    transcript: str = Field(description="Verbatim transcript of audio or cleaned input text.")
    language: str = Field(default="en", description="Detected language code, e.g. en, ta, hi.")
    incident_type: Optional[IncidentType] = Field(
        default=None,
        description="One of: cardiac_arrest, major_trauma, minor_injury, building_collapse, fire, evacuation. Null if undetermined."
    )
    severity: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Severity rating 1 (minor) to 5 (life-threatening/multiple casualties). Null if unknown."
    )
    people_affected: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated number of victims/affected persons. Null if unspecified."
    )
    location_text: str = Field(
        default="",
        description="Exact location, street name, building, or landmark described in the report."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model confidence assessment on extracted facts (0.0 to 1.0)."
    )
    notes: str = Field(
        default="",
        description="Concise operational notes or English translation of key facts."
    )


class IntakeResult(BaseModel):
    """Final output schema including coordinates and verification flags."""
    transcript: str
    language: str = "en"
    incident_type: Optional[IncidentType] = None
    severity: Optional[int] = None
    people_affected: Optional[int] = None
    location_text: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_confirmation: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# System Prompt & Keyword Mappings
# ---------------------------------------------------------------------------

INTAKE_SYSTEM_INSTRUCTION = """You are Crisis Command's intake triaging AI.
Your role is to extract structured emergency incident facts from operator text or audio dispatches.

OPERATING RULES:
1. Extract ONLY information directly stated. NEVER guess, hallucinate, or assume missing parameters.
2. If an attribute is unknown or ambiguous, return null.
3. Map incidents strictly to this IncidentType enum:
   - cardiac_arrest: heart attack, chest pain collapse, unresponsive, no pulse.
   - major_trauma: head injury, severe road accident, bleeding, fractures, vehicle crash.
   - minor_injury: cuts, sprains, superficial wounds.
   - building_collapse: structural failure, trapped under debris.
   - fire: structure fire, smoke, explosion, chemical blaze.
   - evacuation: rising floodwaters, gas leak evacuation, crowd hazard.
4. Severity scale (1 to 5):
   - 5: Life-threatening, cardiac arrest, asphyxiation, active severe arterial bleed, or massive casualties.
   - 4: Serious trauma, multiple injured persons, spreading building fire.
   - 3: Moderate injuries, localized fire, structural instability.
   - 2: Minor injuries with complications.
   - 1: Minor abrasions or non-urgent assistance.
5. Transcribe audio faithfully in its original language (e.g., Tamil, Hindi, English). In the 'notes' field, provide a clean, 1-sentence English translation and operational summary.
6. Extract the precise location landmark or street name into 'location_text' without extra commentary.
"""

# Multilingual keywords for offline fallback (English, Tamil transliterated/Tamil script, Hindi)
MOCK_KEYWORD_MAP = [
    (
        IncidentType.cardiac_arrest,
        5,
        [
            "cardiac", "heart attack", "chest pain", "unconscious", "pulse",
            "மாரடைப்பு", "நெஞ்சு வலி", "மயக்கம்", "दिल का दौरा", "बेहोश", "maradaipu"
        ]
    ),
    (
        IncidentType.building_collapse,
        5,
        [
            "collapse", "debris", "rubble", "trapped", "building fell",
            "கட்டிடம் இடிந்து", "இடிபாடு", "इमारत गिर गई", "मलबे", "idinthu"
        ]
    ),
    (
        IncidentType.fire,
        4,
        [
            "fire", "smoke", "blaze", "flames", "burning",
            "தீ", "நெருப்பு", "புகை", "आग", "धुआं", "thee", "neruppu"
        ]
    ),
    (
        IncidentType.major_trauma,
        4,
        [
            "accident", "crash", "collision", "bleeding", "fracture", "run over",
            "விபத்து", "ரத்தம்", "மோதி", "दुर्घटना", "हादसा", "खून", "vibathu", "ratham"
        ]
    ),
    (
        IncidentType.evacuation,
        3,
        [
            "evacuate", "flood", "submerged", "rising water", "water logging",
            "வெள்ளம்", "வெளியேற்ற", "बाढ़", "पानी भर गया", "vellam"
        ]
    ),
    (
        IncidentType.minor_injury,
        1,
        [
            "minor", "scratch", "small cut", "sprain", "bandage",
            "சிறு காயம்", "காயம்", "चोट", "हल्की चोट", "kaayam"
        ]
    ),
]


# ---------------------------------------------------------------------------
# Offline Mock Extraction
# ---------------------------------------------------------------------------

def _mock_extract(text: str) -> RawIntakeSchema:
    """Offline keyword-based heuristic parsing for hackathon resilience."""
    cleaned = text.lower()
    detected_type: Optional[IncidentType] = None
    detected_severity: Optional[int] = None

    for inc_type, base_sev, keywords in MOCK_KEYWORD_MAP:
        if any(kw in cleaned for kw in keywords):
            detected_type = inc_type
            detected_severity = base_sev
            break

    # Look for casualty numbers like "2 people", "3 injured", "5 persons"
    people_match = re.search(r"(\d+)\s*(people|persons|injured|victims|casualties|பேர்|लोग)", cleaned)
    people_affected = int(people_match.group(1)) if people_match else 1

    # Extract location snippet heuristics: "near ...", "at ...", "in ..."
    loc_match = re.search(r"(?:near|at|around|in front of|close to|பக்கத்தில்|அருகில்)\s+([a-zA-Z0-9\s\.\-]{3,35})", text, re.IGNORECASE)
    loc_text = loc_match.group(1).strip() if loc_match else text[:50].strip()

    confidence = 0.75 if detected_type else 0.40

    return RawIntakeSchema(
        transcript=text.strip(),
        language="ta" if re.search(r"[\u0B80-\u0BFF]", text) else ("hi" if re.search(r"[\u0900-\u097F]", text) else "en"),
        incident_type=detected_type,
        severity=detected_severity or 3,
        people_affected=people_affected,
        location_text=loc_text,
        confidence=confidence,
        notes=f"Extracted via offline fallback parser. Identified keywords for: {detected_type.value if detected_type else 'unknown'}."
    )


# ---------------------------------------------------------------------------
# Gemini Extraction via google-genai SDK
# ---------------------------------------------------------------------------

async def _gemini_extract_call(
    client: genai.Client,
    contents: list,
) -> RawIntakeSchema:
    """Calls Gemini with enforced JSON structured output matching RawIntakeSchema."""
    config = types.GenerateContentConfig(
        system_instruction=INTAKE_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=RawIntakeSchema,
        temperature=0.1,
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=settings.gemini_model,
        contents=contents,
        config=config,
    )

    # In google-genai SDK with response_schema, parsed object is available on response.parsed
    if response.parsed:
        return response.parsed  # type: ignore[return-value]

    # Fallback to direct json parsing of response.text if parsed is empty
    return RawIntakeSchema.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Main Extraction Function
# ---------------------------------------------------------------------------

async def extract(
    text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> IntakeResult:
    """Extracts structured incident data from text or audio dispatch.

    1. Checks LLM_MOCK; if active, performs rule-based parsing.
    2. Otherwise invokes Gemini via google-genai SDK with a 15s timeout and 1 retry.
    3. Resolves geographic coordinates via backend/geocode.py.
    4. Computes needs_confirmation flag and logs an event to the Store.
    """
    raw_schema: Optional[RawIntakeSchema] = None
    input_desc = "Audio recording" if audio_bytes else f"'{text[:60]}...'" if text else "Empty dispatch"

    if settings.llm_mock or not settings.gemini_api_key:
        fallback_text = text or "Emergency reported via voice dispatch near Cantonment Central Bus Stand"
        raw_schema = _mock_extract(fallback_text)
    else:
        client = genai.Client(api_key=settings.gemini_api_key)

        # Build contents payload
        contents: list = []
        if audio_bytes:
            safe_mime = mime_type or "audio/webm"
            # Normalize webm codec strings like audio/webm;codecs=opus -> audio/webm
            clean_mime = safe_mime.split(";")[0].strip()
            contents.append(
                types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
            )
            contents.append("Listen carefully, transcribe this emergency call, and extract the structured triage data.")
        elif text:
            contents.append(text)
        else:
            raw_schema = RawIntakeSchema(
                transcript="",
                location_text="",
                confidence=0.0,
                notes="Received empty dispatch payload."
            )

        if raw_schema is None:
            # 15s timeout with 1 retry
            attempts = 2
            for attempt in range(1, attempts + 1):
                try:
                    raw_schema = await asyncio.wait_for(
                        _gemini_extract_call(client, contents),
                        timeout=15.0,
                    )
                    break
                except Exception as exc:
                    logger.warning("Gemini intake attempt %d failed: %s", attempt, exc)
                    if attempt == attempts:
                        logger.error("All Gemini attempts failed. Falling back to offline mock.")
                        fallback_text = text or "Emergency incident reported via voice dispatch"
                        raw_schema = _mock_extract(fallback_text)
                    else:
                        await asyncio.sleep(0.5)

    assert raw_schema is not None

    # Geocode resolution
    geo_res = await geocode_resolve(raw_schema.location_text)
    lat = geo_res["lat"] if geo_res else None
    lng = geo_res["lng"] if geo_res else None

    # Confidence calculation: factor in geocoding success
    final_conf = raw_schema.confidence
    if not geo_res:
        final_conf = min(final_conf, 0.5)
    elif geo_res.get("source") == "landmark":
        final_conf = max(final_conf, 0.85)

    # Confirmation criteria: low confidence, unresolved location, or missing incident type
    needs_confirm = (
        final_conf < 0.70
        or lat is None
        or lng is None
        or raw_schema.incident_type is None
    )

    result = IntakeResult(
        transcript=raw_schema.transcript,
        language=raw_schema.language,
        incident_type=raw_schema.incident_type,
        severity=raw_schema.severity,
        people_affected=raw_schema.people_affected,
        location_text=raw_schema.location_text,
        lat=lat,
        lng=lng,
        confidence=round(final_conf, 2),
        needs_confirmation=needs_confirm,
        notes=raw_schema.notes,
    )

    # Log summary to Store
    type_str = result.incident_type.value if result.incident_type else "unclassified"
    loc_str = result.location_text or "unknown location"
    coord_str = f"({result.lat:.4f}, {result.lng:.4f})" if result.lat and result.lng else "unresolved coords"
    store.log_event(
        "intake",
        f"Processed {input_desc} -> Type: {type_str}, Sev: {result.severity}, Conf: {result.confidence} at '{loc_str}' {coord_str}."
    )

    return result