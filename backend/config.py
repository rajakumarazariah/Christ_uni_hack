"""backend/config.py
Configuration loader and settings definition for Crisis Command.
Loads environment variables via python-dotenv and exposes a typed, frozen Settings dataclass.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

# Set up basic logging before settings evaluation
logger = logging.getLogger("crisis_command.config")

# Project root directory: crisis-command/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from project root if it exists
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path)


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        logger.warning("Invalid integer for %s: '%s'. Using default %d.", key, val, default)
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        logger.warning("Invalid float for %s: '%s'. Using default %f.", key, val, default)
        return default


def _get_bbox(key: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    val = os.getenv(key)
    if not val or not val.strip():
        return default
    try:
        # Expected format: south,west,north,east
        parts = [float(p.strip()) for p in val.split(",")]
        if len(parts) == 4:
            return (parts[0], parts[1], parts[2], parts[3])
        logger.warning("Invalid bbox format for %s: '%s'. Expected 4 comma-separated floats.", key, val)
        return default
    except Exception:
        logger.warning("Failed to parse bbox %s: '%s'. Falling back to default.", key, val)
        return default


@dataclass(frozen=True)
class Settings:
    # Gemini
    gemini_api_key: str
    gemini_model: str
    llm_mock: bool

    # Routing / geocoding
    use_osrm: bool
    osrm_base_url: str
    nominatim_url: str
    nominatim_user_agent: str

    # Demo city
    demo_city_name: str
    demo_center_lat: float
    demo_center_lng: float
    demo_bbox: Tuple[float, float, float, float]  # (south, west, north, east)

    # Solver / simulation
    target_response_min: int
    switch_penalty: float
    avg_speed_kmph: float
    sim_speed: int

    # Server
    host: str
    port: int
    log_level: str


def _load_settings() -> Settings:
    raw_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    mock_flag = _get_bool("LLM_MOCK", False)

    # If mock mode is off but no API key is provided, log warning and force mock mode on
    if not mock_flag and not raw_key:
        logger.warning("GEMINI_API_KEY is missing or empty. Forcing LLM_MOCK=True.")
        mock_flag = True

    return Settings(
        # Gemini
        gemini_api_key=raw_key,
        gemini_model=model,
        llm_mock=mock_flag,
        # Routing / geocoding
        use_osrm=_get_bool("USE_OSRM", True),
        osrm_base_url=os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/"),
        nominatim_url=os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org").rstrip("/"),
        nominatim_user_agent=os.getenv(
            "NOMINATIM_USER_AGENT",
            "crisis-command-hackathon (dispatch@crisiscommand.local)"
        ),
        # Demo city (Default: Tiruchirappalli)
        demo_city_name=os.getenv("DEMO_CITY_NAME", "Tiruchirappalli"),
        demo_center_lat=_get_float("DEMO_CENTER_LAT", 10.8050),
        demo_center_lng=_get_float("DEMO_CENTER_LNG", 78.6856),
        demo_bbox=_get_bbox("DEMO_BBOX", (10.72, 78.60, 10.90, 78.78)),
        # Solver / simulation
        target_response_min=_get_int("TARGET_RESPONSE_MIN", 10),
        switch_penalty=_get_float("SWITCH_PENALTY", 0.15),
        avg_speed_kmph=_get_float("AVG_SPEED_KMPH", 40.0),
        sim_speed=_get_int("SIM_SPEED", 30),
        # Server
        host=os.getenv("HOST", "127.0.0.1"),
        port=_get_int("PORT", 8000),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


# Module-level singleton
settings: Settings = _load_settings()


def data_path(name: str) -> Path:
    """Returns the absolute path to a file inside backend/data/."""
    return Path(__file__).resolve().parent / "data" / name