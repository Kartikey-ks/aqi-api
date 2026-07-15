"""
AeroGuard – Unified FastAPI Backend
====================================
Combines:
  • Kartik's FastAPI backend (data ingestion, XGBoost forecasting, ward endpoints)
  • LangGraph Multi-Agent AI Engine (anomaly detection → source attribution → enforcement)

Stage 2 + Stage 3 of the AeroGuard architecture.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import requests
import time
import json
from datetime import datetime
import os
from pathlib import Path
from shapely.geometry import shape, Point
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Import the LangGraph multi-agent engine
from ai_engine import aeroguard_app, CityState

app = FastAPI(
    title="AeroGuard – Delhi Air Quality Command Center",
    description="Unified backend: live AQI data + XGBoost forecasting + LangGraph AI agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Base paths (all data files sit alongside this script)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load XGBoost models (v2: trained with weather + station_id features)
# ---------------------------------------------------------------------------
models = {}
for horizon in ["target_1day", "target_2day", "target_3day"]:
    m = XGBRegressor()
    m.load_model(str(BASE_DIR / "models" / f"{horizon}_model.json"))
    models[horizon] = m

FEATURE_COLS = [
    "pm25", "pm25_lag1", "pm25_lag2", "pm25_lag7", "pm25_roll7",
    "day_of_week", "month", "station_id",
    "temp_mean", "humidity_mean", "windspeed_mean", "winddirection_mean", "precipitation_sum"
]

# ---------------------------------------------------------------------------
# Load historical data
# ---------------------------------------------------------------------------
historical_df = pd.read_csv(str(BASE_DIR / "delhi_pm25_historical_v2.csv"))
historical_df["date"] = pd.to_datetime(historical_df["date"])


def normalize_name(name):
    return name.strip().lower().replace("new delhi", "delhi")


historical_df["name_normalized"] = historical_df["location_name"].apply(normalize_name)

with open(str(BASE_DIR / "models" / "station_ids.json")) as f:
    STATION_IDS_RAW = json.load(f)

STATION_IDS = {normalize_name(name): sid for name, sid in STATION_IDS_RAW.items()}

# ---------------------------------------------------------------------------
# Delhi district boundaries (for ward/zone aggregation)
# ---------------------------------------------------------------------------
with open(str(BASE_DIR / "delhi_districts.geojson")) as f:
    DISTRICTS_GEOJSON = json.load(f)

DISTRICT_POLYGONS = [
    (feat["properties"]["dtname"], shape(feat["geometry"]))
    for feat in DISTRICTS_GEOJSON["features"]
]


def find_district(lat, lon):
    """Point-in-polygon lookup."""
    pt = Point(lon, lat)
    for name, poly in DISTRICT_POLYGONS:
        if poly.contains(pt):
            return name
    return None


# ---------------------------------------------------------------------------
# Load mock citizen grievances
# ---------------------------------------------------------------------------
with open(str(BASE_DIR / "mock_grievances.json")) as f:
    MOCK_GRIEVANCES = json.load(f)

# ---------------------------------------------------------------------------
# OpenAQ setup (live PM2.5)
# ---------------------------------------------------------------------------
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")
HEADERS = {"X-API-Key": OPENAQ_API_KEY}

STARTUP_ERROR = None


def get_delhi_pm25_locations():
    global STARTUP_ERROR
    url = "https://api.openaq.org/v3/locations"
    params = {"coordinates": "28.6139,77.2090", "radius": 20000, "limit": 100}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    except Exception as e:
        STARTUP_ERROR = f"Request failed: {e}"
        return []

    if resp.status_code != 200:
        STARTUP_ERROR = f"Status {resp.status_code}: {resp.text[:500]}"
        return []

    data = resp.json()
    if "results" not in data:
        STARTUP_ERROR = f"No 'results' key. Raw response: {data}"
        return []

    locations = []
    for loc in data["results"]:
        pm25_sensor_id = None
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("name") == "pm25":
                pm25_sensor_id = sensor["id"]
                break
        if pm25_sensor_id is None:
            continue

        lat = loc["coordinates"]["latitude"]
        lon = loc["coordinates"]["longitude"]

        locations.append({
            "location_id": loc["id"],
            "location_name": loc["name"],
            "pm25_sensor_id": pm25_sensor_id,
            "lat": lat,
            "lon": lon,
            "district": find_district(lat, lon)
        })
    return locations


try:
    PM25_LOCATIONS = get_delhi_pm25_locations()
except Exception as e:
    STARTUP_ERROR = f"Unhandled exception: {e}"
    PM25_LOCATIONS = []

# ---------------------------------------------------------------------------
# Weather (Open-Meteo, free, no key) - city-wide, same as training
# ---------------------------------------------------------------------------
_weather_cache = {"data": None, "timestamp": None}
WEATHER_CACHE_TTL_SECONDS = 60 * 60


def fetch_current_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,precipitation",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    wdf = pd.DataFrame(hourly)

    return {
        "temp_mean": float(wdf["temperature_2m"].mean()),
        "humidity_mean": float(wdf["relativehumidity_2m"].mean()),
        "windspeed_mean": float(wdf["windspeed_10m"].mean()),
        "winddirection_mean": float(wdf["winddirection_10m"].mean()),
        "precipitation_sum": float(wdf["precipitation"].sum())
    }


def get_current_weather_cached():
    now = datetime.now()
    if (
        _weather_cache["data"] is not None
        and (now - _weather_cache["timestamp"]).total_seconds() < WEATHER_CACHE_TTL_SECONDS
    ):
        return _weather_cache["data"]

    try:
        data = fetch_current_weather()
        _weather_cache["data"] = data
        _weather_cache["timestamp"] = now
        return data
    except Exception:
        if _weather_cache["data"] is not None:
            return _weather_cache["data"]
        raise


def fetch_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "daily": "temperature_2m_max,temperature_2m_min,windspeed_10m_max,winddirection_10m_dominant,precipitation_sum",
        "timezone": "Asia/Kolkata",
        "forecast_days": 4
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["daily"]


# ---------------------------------------------------------------------------
# AQI category + rule-based action recommendations
# ---------------------------------------------------------------------------
def aqi_category(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Moderate"
    elif value <= 200:
        return "Poor"
    elif value <= 300:
        return "Severe"
    else:
        return "Hazardous"


def recommended_actions(current_pm25, forecast_1day):
    worst = max(current_pm25, forecast_1day)
    actions = []

    if worst > 300:
        actions = [
            "Implement odd-even vehicle scheme",
            "Halt all non-essential construction activity",
            "Suspend outdoor school activities; consider temporary closure",
            "Deploy anti-smog guns / water sprinkling in hotspot zones",
            "Issue public health advisory for outdoor exposure"
        ]
    elif worst > 200:
        actions = [
            "Halt construction and demolition activity",
            "Restrict entry of polluting/older diesel vehicles",
            "Increase water sprinkling on major roads",
            "Advisory for outdoor workers and vulnerable groups"
        ]
    elif worst > 100:
        actions = [
            "Monitor closely; prepare contingency measures",
            "Advisory for sensitive groups (elderly, children, respiratory conditions)"
        ]
    else:
        actions = ["No immediate action required"]

    return actions


# ---------------------------------------------------------------------------
# Wind direction helpers
# ---------------------------------------------------------------------------
COMPASS_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]


def degrees_to_compass(deg):
    """Convert wind direction in degrees to compass direction string."""
    idx = round(deg / 22.5) % 16
    return COMPASS_DIRECTIONS[idx]


# ---------------------------------------------------------------------------
# Live+forecast cache
# ---------------------------------------------------------------------------
_cache = {"data": None, "timestamp": None}
CACHE_TTL_SECONDS = 15 * 60


class ForecastInput(BaseModel):
    pm25: float
    pm25_lag1: float
    pm25_lag2: float
    pm25_lag7: float
    pm25_roll7: float
    day_of_week: int
    month: int
    station_id: int
    temp_mean: float
    humidity_mean: float
    windspeed_mean: float
    winddirection_mean: float
    precipitation_sum: float


# ===================================================================
#  ENDPOINTS – Kartik's original endpoints (Stage 2)
# ===================================================================

@app.get("/")
def home():
    return {"status": "AeroGuard API is running", "version": "1.0.0"}


@app.post("/forecast")
def forecast(data: ForecastInput):
    features = np.array([[getattr(data, col) for col in FEATURE_COLS]])
    predictions = {}
    for horizon, model in models.items():
        pred = max(0.0, float(model.predict(features)[0]))
        predictions[horizon] = round(pred, 2)

    return {
        "input": data.dict(),
        "forecast": {
            "1_day": predictions["target_1day"],
            "2_day": predictions["target_2day"],
            "3_day": predictions["target_3day"]
        }
    }


@app.get("/weather")
def weather():
    try:
        current = get_current_weather_cached()
    except Exception as e:
        return {"error": f"Failed to fetch current weather: {e}"}

    try:
        forecast_daily = fetch_weather_forecast()
    except Exception:
        forecast_daily = None

    return {"current": current, "forecast_daily": forecast_daily}


def _build_station_forecasts():
    """Per-station logic: fetch live PM2.5, build features, predict."""
    results = []
    skipped = []

    try:
        weather_now = get_current_weather_cached()
    except Exception as e:
        return [], [{"reason": f"weather fetch failed: {e}"}], None

    for loc in PM25_LOCATIONS:
        time.sleep(0.5)

        url = f"https://api.openaq.org/v3/locations/{loc['location_id']}/latest"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
        except Exception as e:
            skipped.append({"location": loc["location_name"], "reason": f"request error: {e}"})
            continue

        if resp.status_code == 429:
            time.sleep(2)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
            except Exception as e:
                skipped.append({"location": loc["location_name"], "reason": f"retry request error: {e}"})
                continue

        if resp.status_code != 200:
            skipped.append({"location": loc["location_name"], "reason": f"status {resp.status_code}"})
            continue

        latest_results = resp.json().get("results", [])
        current_pm25 = None
        for r in latest_results:
            if r.get("sensorsId") == loc["pm25_sensor_id"]:
                current_pm25 = r.get("value")
                break

        if current_pm25 is None or current_pm25 <= 0:
            skipped.append({"location": loc["location_name"], "reason": f"invalid reading: {current_pm25}"})
            continue

        loc_normalized = normalize_name(loc["location_name"])
        station_id = STATION_IDS.get(loc_normalized)
        if station_id is None:
            skipped.append({"location": loc["location_name"], "reason": "no station_id mapping"})
            continue

        hist = historical_df[historical_df["name_normalized"] == loc_normalized].sort_values("date")
        if len(hist) < 7:
            skipped.append({"location": loc["location_name"], "reason": f"only {len(hist)} historical rows"})
            continue

        if loc["district"] is None:
            skipped.append({"location": loc["location_name"], "reason": "outside Delhi district boundaries (likely NCR: Noida/Ghaziabad)"})
            continue

        lag1 = hist["pm25"].iloc[-1]
        lag2 = hist["pm25"].iloc[-2]
        lag7 = hist["pm25"].iloc[-7]
        roll7 = hist["pm25"].tail(7).mean()

        today = datetime.now()
        row = {
            "pm25": current_pm25, "pm25_lag1": lag1, "pm25_lag2": lag2,
            "pm25_lag7": lag7, "pm25_roll7": roll7,
            "day_of_week": today.weekday(), "month": today.month,
            "station_id": station_id,
            "temp_mean": weather_now["temp_mean"], "humidity_mean": weather_now["humidity_mean"],
            "windspeed_mean": weather_now["windspeed_mean"], "winddirection_mean": weather_now["winddirection_mean"],
            "precipitation_sum": weather_now["precipitation_sum"],
        }
        features = np.array([[row[col] for col in FEATURE_COLS]])

        forecast_vals = {}
        for horizon, model in models.items():
            pred = max(0.0, float(model.predict(features)[0]))
            forecast_vals[horizon] = round(pred, 2)

        results.append({
            "name": loc["location_name"],
            "district": loc["district"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "current_pm25": current_pm25,
            "forecast": {
                "1_day": forecast_vals["target_1day"],
                "2_day": forecast_vals["target_2day"],
                "3_day": forecast_vals["target_3day"]
            }
        })

    return results, skipped, weather_now


def _build_ward_forecast():
    stations, skipped, weather_now = _build_station_forecasts()

    by_district = {}
    for s in stations:
        by_district.setdefault(s["district"], []).append(s)

    wards = []
    for district_name, station_list in by_district.items():
        avg_current = round(np.mean([s["current_pm25"] for s in station_list]), 2)
        avg_1day = round(np.mean([s["forecast"]["1_day"] for s in station_list]), 2)
        avg_2day = round(np.mean([s["forecast"]["2_day"] for s in station_list]), 2)
        avg_3day = round(np.mean([s["forecast"]["3_day"] for s in station_list]), 2)

        wards.append({
            "district": district_name,
            "station_count": len(station_list),
            "stations": [s["name"] for s in station_list],
            "current_pm25": avg_current,
            "current_category": aqi_category(avg_current),
            "forecast": {
                "1_day": avg_1day,
                "2_day": avg_2day,
                "3_day": avg_3day
            },
            "forecast_1day_category": aqi_category(avg_1day),
            "recommended_actions": recommended_actions(avg_current, avg_1day)
        })

    covered_names = {w["district"] for w in wards}
    all_district_names = {name for name, _ in DISTRICT_POLYGONS}
    for name in all_district_names - covered_names:
        wards.append({
            "district": name,
            "station_count": 0,
            "stations": [],
            "current_pm25": None,
            "current_category": "No data",
            "forecast": {"1_day": None, "2_day": None, "3_day": None},
            "forecast_1day_category": "No data",
            "recommended_actions": ["No sensor coverage in this district"]
        })

    return {
        "wards": wards,
        "station_level_skipped": skipped,
        "skipped_count": len(skipped),
        "weather_used": weather_now
    }


@app.get("/ward-forecast")
def ward_forecast(refresh: bool = False):
    now = datetime.now()
    if (
        not refresh
        and _cache["data"] is not None
        and (now - _cache["timestamp"]).total_seconds() < CACHE_TTL_SECONDS
    ):
        return _cache["data"]

    data = _build_ward_forecast()
    _cache["data"] = data
    _cache["timestamp"] = now
    return data


@app.get("/district-boundaries")
def district_boundaries():
    """Serves the raw GeoJSON so the frontend can draw district polygons."""
    return DISTRICTS_GEOJSON


@app.get("/debug")
def debug():
    debug_info = {
        "openaq_key_present": bool(OPENAQ_API_KEY),
        "groq_key_present": bool(os.environ.get("GROQ_API_KEY")),
        "startup_error": STARTUP_ERROR,
        "num_locations_found": len(PM25_LOCATIONS),
        "num_districts_loaded": len(DISTRICT_POLYGONS),
        "num_grievances_loaded": len(MOCK_GRIEVANCES),
        "sample_locations_with_district": PM25_LOCATIONS[:5],
    }
    try:
        debug_info["current_weather"] = get_current_weather_cached()
    except Exception as e:
        debug_info["weather_error"] = str(e)
    return debug_info


# ===================================================================
#  NEW ENDPOINT – /analyze-city  (Stage 2 → Stage 3 integration)
# ===================================================================
#  This is the KEY endpoint from the architecture diagram.
#  Frontend clicks a district → GET /analyze-city?district=East
#  Backend:
#    1. Gathers sensor data for that district
#    2. Fetches weather (wind direction/speed)
#    3. Checks AQI threshold
#    4. If hazardous → invokes LangGraph pipeline
#    5. Returns unified JSON (forecast + AI analysis + dispatch order)
# ===================================================================

@app.get("/analyze-city")
def analyze_city(district: str = "Delhi", city: str = None):
    """
    Full AeroGuard pipeline for a specific district/city.

    Query params:
      - district: District name (e.g. "East", "Central", "North West")
      - city: Alias for district (for backward compatibility with architecture diagram)

    Returns:
      - sensor_data: Live station readings + forecasts
      - weather: Current weather conditions
      - ai_analysis: LangGraph multi-agent results (anomaly, source, enforcement)
      - public_advisory: Citizen-facing advisory message
    """
    # Allow 'city' param as alias
    target = city if city else district

    # ---- Step 1: Get sensor data for the target district ----
    try:
        weather_now = get_current_weather_cached()
    except Exception as e:
        return {"error": f"Failed to fetch weather: {e}"}

    # Find matching stations in this district
    district_stations = []
    for loc in PM25_LOCATIONS:
        if loc["district"] and loc["district"].lower() == target.lower():
            district_stations.append(loc)

    # If no stations match by district, try city-wide
    if not district_stations:
        district_stations = [loc for loc in PM25_LOCATIONS if loc["district"] is not None]
        if not district_stations:
            return {
                "error": f"No sensor stations found for '{target}'",
                "available_districts": list({loc["district"] for loc in PM25_LOCATIONS if loc["district"]})
            }

    # Build sensor data array for the AI pipeline
    sensor_data_for_ai = []
    station_forecasts = []

    for loc in district_stations[:5]:  # Limit to avoid rate limiting
        time.sleep(0.3)
        url = f"https://api.openaq.org/v3/locations/{loc['location_id']}/latest"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        latest_results = resp.json().get("results", [])
        current_pm25 = None
        for r in latest_results:
            if r.get("sensorsId") == loc["pm25_sensor_id"]:
                current_pm25 = r.get("value")
                break

        if current_pm25 is None or current_pm25 <= 0:
            continue

        sensor_entry = {
            "station": loc["location_name"],
            "AQI": round(current_pm25 * 1.5),  # Approximate AQI from PM2.5
            "PM2_5": round(current_pm25, 1),
            "PM10": round(current_pm25 * 1.6, 1),  # Approximate PM10
        }
        sensor_data_for_ai.append(sensor_entry)

        # Also build XGBoost forecast for this station
        loc_normalized = normalize_name(loc["location_name"])
        station_id = STATION_IDS.get(loc_normalized)
        if station_id is not None:
            hist = historical_df[historical_df["name_normalized"] == loc_normalized].sort_values("date")
            if len(hist) >= 7:
                lag1 = hist["pm25"].iloc[-1]
                lag2 = hist["pm25"].iloc[-2]
                lag7 = hist["pm25"].iloc[-7]
                roll7 = hist["pm25"].tail(7).mean()
                today = datetime.now()
                row = {
                    "pm25": current_pm25, "pm25_lag1": lag1, "pm25_lag2": lag2,
                    "pm25_lag7": lag7, "pm25_roll7": roll7,
                    "day_of_week": today.weekday(), "month": today.month,
                    "station_id": station_id,
                    **weather_now
                }
                features = np.array([[row[col] for col in FEATURE_COLS]])
                forecast_vals = {}
                for horizon, model in models.items():
                    pred = max(0.0, float(model.predict(features)[0]))
                    forecast_vals[horizon] = round(pred, 2)

                station_forecasts.append({
                    "station": loc["location_name"],
                    "current_pm25": current_pm25,
                    "forecast": {
                        "1_day": forecast_vals["target_1day"],
                        "2_day": forecast_vals["target_2day"],
                        "3_day": forecast_vals["target_3day"],
                    }
                })

    if not sensor_data_for_ai:
        return {
            "error": "Could not fetch any live sensor data for this district",
            "target": target,
            "weather": weather_now
        }

    # ---- Step 2: Determine if hazardous ----
    max_aqi = max(s["AQI"] for s in sensor_data_for_ai)
    is_hazardous = max_aqi > 200

    # ---- Step 3: Get relevant grievances for this district ----
    district_grievances = [
        g for g in MOCK_GRIEVANCES
        if g["district"].lower() == target.lower()
    ]
    # If no district-specific grievances, use all as context
    if not district_grievances:
        district_grievances = MOCK_GRIEVANCES[:5]

    # ---- Step 4: Compute wind direction from weather ----
    wind_direction = degrees_to_compass(weather_now["winddirection_mean"])
    wind_speed = round(weather_now["windspeed_mean"], 1)

    # ---- Step 5: Invoke LangGraph pipeline ----
    ai_analysis = None
    if is_hazardous and os.environ.get("GROQ_API_KEY"):
        try:
            langgraph_input: CityState = {
                "target_location": f"{target}, Delhi",
                "sensor_data": sensor_data_for_ai,
                "wind_direction": wind_direction,
                "wind_speed_kmh": wind_speed,
                "regional_grievances": district_grievances,
                "is_hazardous": False,  # Let the anomaly detector decide
                "anomaly_summary": "",
                "identified_source": "",
                "enforcement_order": {},
            }

            result = aeroguard_app.invoke(langgraph_input)

            ai_analysis = {
                "is_hazardous": result["is_hazardous"],
                "anomaly_summary": result["anomaly_summary"],
                "identified_source": result.get("identified_source", "N/A"),
                "enforcement_order": result.get("enforcement_order", {}),
            }
        except Exception as e:
            ai_analysis = {
                "error": f"AI pipeline failed: {str(e)}",
                "is_hazardous": is_hazardous,
                "anomaly_summary": f"Max AQI of {max_aqi} detected – exceeds safe threshold",
            }
    else:
        # No GROQ key or not hazardous – provide rule-based analysis
        avg_pm25 = np.mean([s["PM2_5"] for s in sensor_data_for_ai])
        ai_analysis = {
            "is_hazardous": is_hazardous,
            "anomaly_summary": (
                f"Max AQI of {max_aqi} detected. "
                + ("HAZARDOUS – immediate action required." if is_hazardous
                   else "Within safe limits – monitoring continues.")
            ),
            "identified_source": "N/A" if not is_hazardous else "Rule-based: multiple sources likely (AI analysis requires GROQ_API_KEY)",
            "enforcement_order": {} if not is_hazardous else {
                "Action_Type": "Multi-department coordination required",
                "Assigned_Department": "Environmental Office",
                "Priority_Level": "Critical" if max_aqi > 300 else "High",
                "Justification": f"AQI {max_aqi} exceeds safe threshold in {target}, Delhi"
            },
        }

    # ---- Step 6: Build public advisory ----
    avg_pm25 = np.mean([s["PM2_5"] for s in sensor_data_for_ai])
    forecast_1d = station_forecasts[0]["forecast"]["1_day"] if station_forecasts else avg_pm25
    actions = recommended_actions(avg_pm25, forecast_1d)

    public_advisory = {
        "severity": aqi_category(max(avg_pm25, forecast_1d)),
        "message": _generate_advisory_message(target, avg_pm25, forecast_1d, is_hazardous),
        "recommended_actions": actions,
    }

    # ---- Final Response ----
    return {
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "sensor_data": sensor_data_for_ai,
        "station_forecasts": station_forecasts,
        "weather": {
            "temperature": weather_now["temp_mean"],
            "humidity": weather_now["humidity_mean"],
            "wind_speed_kmh": wind_speed,
            "wind_direction": wind_direction,
            "wind_direction_degrees": weather_now["winddirection_mean"],
            "precipitation_sum": weather_now["precipitation_sum"],
        },
        "ai_analysis": ai_analysis,
        "grievances_considered": district_grievances,
        "public_advisory": public_advisory,
    }


def _generate_advisory_message(district, current_pm25, forecast_1d, is_hazardous):
    """Generate a citizen-facing advisory message."""
    if is_hazardous:
        return (
            f"⚠️ HEALTH ALERT for {district}, Delhi: "
            f"Air quality has reached hazardous levels (PM2.5: {current_pm25:.0f} µg/m³). "
            f"Avoid outdoor activities. Keep windows closed. Use N95 masks if going outside. "
            f"Children, elderly, and those with respiratory conditions should stay indoors. "
            f"Forecast indicates PM2.5 of {forecast_1d:.0f} µg/m³ tomorrow."
        )
    elif current_pm25 > 100:
        return (
            f"🟠 AIR QUALITY ADVISORY for {district}, Delhi: "
            f"PM2.5 levels are elevated at {current_pm25:.0f} µg/m³. "
            f"Sensitive groups should limit prolonged outdoor exertion. "
            f"Tomorrow's forecast: {forecast_1d:.0f} µg/m³."
        )
    else:
        return (
            f"🟢 Air quality in {district}, Delhi is currently within acceptable limits "
            f"(PM2.5: {current_pm25:.0f} µg/m³). No special precautions needed."
        )


# ===================================================================
#  Grievances endpoint
# ===================================================================

@app.get("/grievances")
def get_grievances(district: str = None):
    """Return mock citizen grievances, optionally filtered by district."""
    if district:
        filtered = [g for g in MOCK_GRIEVANCES if g["district"].lower() == district.lower()]
        return {"district": district, "grievances": filtered, "total": len(filtered)}
    return {"grievances": MOCK_GRIEVANCES, "total": len(MOCK_GRIEVANCES)}


# ===================================================================
#  Serve frontend static files
# ===================================================================
frontend_dir = BASE_DIR.parent / "frontend"
if frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
