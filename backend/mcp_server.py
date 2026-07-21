"""
AeroGuard – Environmental MCP Server
=====================================
A dedicated MCP (Model Context Protocol) server that acts as the
universal data-access layer for external environmental sources.

Tools:
  • fetch_openaq_live  – Live air quality data from OpenAQ
  • get_wind_patterns  – Atmospheric data from Open-Meteo

Resources:
  • municipal_grievances://{ward_id} – Citizen grievance records

This server is consumed by the LangGraph agents (MCP Client) via
stdio transport.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "AeroGuard Environmental Data",
    instructions=(
        "Provides real-time environmental data for Delhi air quality analysis. "
        "Use fetch_openaq_live to get pollutant readings, get_wind_patterns for "
        "atmospheric conditions, and municipal_grievances resources for citizen complaints."
    ),
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
OPENAQ_HEADERS = {"X-API-Key": OPENAQ_API_KEY}

# Delhi centre coordinates
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

# ---------------------------------------------------------------------------
# Load grievance data at module level
# ---------------------------------------------------------------------------
_grievances_path = BASE_DIR / "mock_grievances.json"
if _grievances_path.exists():
    with open(str(_grievances_path)) as f:
        GRIEVANCES_DATA: list[dict] = json.load(f)
else:
    GRIEVANCES_DATA = []

# ---------------------------------------------------------------------------
# Internal caches (in-process, lightweight)
# ---------------------------------------------------------------------------
_weather_cache: dict = {"data": None, "timestamp": None}
_WEATHER_TTL = 60 * 60  # 1 hour

_openaq_locations_cache: dict = {"data": None, "timestamp": None}
_LOCATIONS_TTL = 60 * 30  # 30 minutes


# ===================================================================
#  TOOL 1: fetch_openaq_live
# ===================================================================

def _get_delhi_locations() -> list[dict]:
    """Fetch (and cache) Delhi PM2.5 monitoring locations from OpenAQ."""
    now = datetime.now()
    if (
        _openaq_locations_cache["data"] is not None
        and _openaq_locations_cache["timestamp"] is not None
        and (now - _openaq_locations_cache["timestamp"]).total_seconds() < _LOCATIONS_TTL
    ):
        return _openaq_locations_cache["data"]

    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{DELHI_LAT},{DELHI_LON}",
        "radius": 20000,
        "limit": 100,
    }

    try:
        resp = requests.get(url, headers=OPENAQ_HEADERS, params=params, timeout=15)
        if resp.status_code != 200:
            return _openaq_locations_cache.get("data") or []
        data = resp.json()
    except Exception:
        return _openaq_locations_cache.get("data") or []

    locations = []
    for loc in data.get("results", []):
        pm25_sensor_id = None
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("name") == "pm25":
                pm25_sensor_id = sensor["id"]
                break
        if pm25_sensor_id is None:
            continue

        locations.append({
            "location_id": loc["id"],
            "location_name": loc["name"],
            "pm25_sensor_id": pm25_sensor_id,
            "lat": loc["coordinates"]["latitude"],
            "lon": loc["coordinates"]["longitude"],
        })

    _openaq_locations_cache["data"] = locations
    _openaq_locations_cache["timestamp"] = now
    return locations


@mcp.tool()
def fetch_openaq_live(city: str, parameter: str = "pm25") -> str:
    """Fetch live air quality readings from OpenAQ for monitoring stations
    near the given city (currently optimised for Delhi).

    Args:
        city: City name (e.g. "Delhi"). Used for context; queries Delhi sensors.
        parameter: Pollutant parameter to fetch. Default "pm25".

    Returns:
        JSON string with an array of station readings, each containing
        station name, latitude, longitude, and the current pollutant value.
    """
    locations = _get_delhi_locations()
    if not locations:
        return json.dumps({
            "error": "No monitoring locations available",
            "city": city,
            "parameter": parameter,
        })

    readings = []
    for loc in locations[:10]:  # Cap to avoid rate-limiting
        time.sleep(0.3)  # Rate-limit guard

        url = f"https://api.openaq.org/v3/locations/{loc['location_id']}/latest"
        try:
            resp = requests.get(url, headers=OPENAQ_HEADERS, timeout=10)

            # Retry once on 429
            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.get(url, headers=OPENAQ_HEADERS, timeout=10)

            if resp.status_code != 200:
                continue
        except Exception:
            continue

        latest_results = resp.json().get("results", [])
        current_value = None
        for r in latest_results:
            if r.get("sensorsId") == loc["pm25_sensor_id"]:
                current_value = r.get("value")
                break

        if current_value is not None and current_value > 0:
            readings.append({
                "station": loc["location_name"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "parameter": parameter,
                "value": round(current_value, 2),
                "AQI_approx": round(current_value * 1.5),  # Approximate AQI
                "PM2_5": round(current_value, 1),
                "PM10_approx": round(current_value * 1.6, 1),
            })

    return json.dumps({
        "city": city,
        "parameter": parameter,
        "timestamp": datetime.now().isoformat(),
        "station_count": len(readings),
        "readings": readings,
    }, indent=2)


# ===================================================================
#  TOOL 2: get_wind_patterns
# ===================================================================

@mcp.tool()
def get_wind_patterns(latitude: float = DELHI_LAT, longitude: float = DELHI_LON) -> str:
    """Fetch current atmospheric/wind conditions from Open-Meteo for
    pollution dispersion analysis.

    Args:
        latitude: Latitude of the location. Defaults to Delhi centre.
        longitude: Longitude of the location. Defaults to Delhi centre.

    Returns:
        JSON string with wind speed (km/h), wind direction (degrees and
        compass), temperature, humidity, and precipitation data.
    """
    now = datetime.now()

    # Check cache
    if (
        _weather_cache["data"] is not None
        and _weather_cache["timestamp"] is not None
        and (now - _weather_cache["timestamp"]).total_seconds() < _WEATHER_TTL
    ):
        return json.dumps(_weather_cache["data"], indent=2)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,precipitation",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except Exception as e:
        if _weather_cache["data"] is not None:
            return json.dumps(_weather_cache["data"], indent=2)
        return json.dumps({"error": f"Failed to fetch weather: {e}"})

    # Compute daily aggregates
    import statistics
    temps = [t for t in hourly["temperature_2m"] if t is not None]
    humids = [h for h in hourly["relativehumidity_2m"] if h is not None]
    winds = [w for w in hourly["windspeed_10m"] if w is not None]
    wind_dirs = [d for d in hourly["winddirection_10m"] if d is not None]
    precip = [p for p in hourly["precipitation"] if p is not None]

    wind_dir_deg = statistics.mean(wind_dirs) if wind_dirs else 0
    wind_speed = statistics.mean(winds) if winds else 0

    # Convert degrees to compass direction
    compass = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    compass_dir = compass[round(wind_dir_deg / 22.5) % 16]

    result = {
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": now.isoformat(),
        "wind_speed_kmh": round(wind_speed, 1),
        "wind_direction_degrees": round(wind_dir_deg, 1),
        "wind_direction_compass": compass_dir,
        "temperature_celsius": round(statistics.mean(temps), 1) if temps else None,
        "humidity_percent": round(statistics.mean(humids), 1) if humids else None,
        "precipitation_sum_mm": round(sum(precip), 2) if precip else 0,
    }

    _weather_cache["data"] = result
    _weather_cache["timestamp"] = now
    return json.dumps(result, indent=2)


# ===================================================================
#  RESOURCE: municipal_grievances://{ward_id}
# ===================================================================

@mcp.resource("municipal_grievances://{ward_id}")
def get_grievances(ward_id: str) -> str:
    """Read citizen grievances for a specific ward/district of Delhi.

    The ward_id corresponds to a Delhi district name (e.g. "East",
    "Central", "North West"). Use "all" to retrieve all grievances.

    Returns:
        JSON string with filtered grievance records.
    """
    if ward_id.lower() == "all":
        filtered = GRIEVANCES_DATA
    else:
        filtered = [
            g for g in GRIEVANCES_DATA
            if g.get("district", "").lower() == ward_id.lower()
        ]

    # If no district-specific grievances found, return all as fallback context
    if not filtered and ward_id.lower() != "all":
        filtered = GRIEVANCES_DATA[:5]

    return json.dumps({
        "ward_id": ward_id,
        "total": len(filtered),
        "grievances": filtered,
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point – stdio transport (invoked as subprocess by MCP client)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
