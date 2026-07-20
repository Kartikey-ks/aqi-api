"""
source_attribution.py
======================
Geospatial source attribution: construction + traffic proximity scoring.

Given a monitoring station's coordinates, this module finds nearby
construction sites and traffic corridors (from land_use_zones.json) and
computes a weighted "contribution score" for each, based on distance and
zone intensity. It then rolls these up into a dominant-source summary
(construction vs traffic vs unknown).

Wind-based directional weighting (upwind triangulation) can be added later
as a multiplier on top of contribution_score, without restructuring this
module — see get_source_attribution() docstring.
"""

import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_land_use_zones():
    with open(str(BASE_DIR / "land_use_zones.json")) as f:
        raw = json.load(f)["features"]
    return [
        {
            "zone_name": z["properties"]["zone_name"],
            "source_type": z["properties"]["source_type"],
            "intensity": z["properties"]["intensity"],
            "lat": z["geometry"]["coordinates"][1],
            "lon": z["geometry"]["coordinates"][0],
        }
        for z in raw
    ]


LAND_USE_ZONES = load_land_use_zones()

_INTENSITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_nearby_sources(station_lat, station_lon, max_radius_km=8):
    """
    Find construction/traffic zones within max_radius_km of a station.
    Each result includes a contribution_score in [0, 1] — higher means
    closer + higher intensity (i.e. more likely to be affecting this station).
    """
    nearby = []

    for zone in LAND_USE_ZONES:
        dist = haversine_km(station_lat, station_lon, zone["lat"], zone["lon"])
        if dist > max_radius_km:
            continue

        proximity_factor = max(0.0, 1 - (dist / max_radius_km))
        weight = _INTENSITY_WEIGHT.get(zone["intensity"], 0.5)
        contribution_score = round(proximity_factor * weight, 2)

        nearby.append({
            "zone_name": zone["zone_name"],
            "source_type": zone["source_type"],
            "distance_km": round(dist, 1),
            "intensity": zone["intensity"],
            "contribution_score": contribution_score,
        })

    nearby.sort(key=lambda x: -x["contribution_score"])
    return nearby


def summarize_source_contribution(nearby_sources):
    """Roll up individual zone scores into a single dominant-source verdict."""
    if not nearby_sources:
        return {"dominant_source": "unknown", "construction_score": 0, "traffic_score": 0}

    construction_score = sum(
        s["contribution_score"] for s in nearby_sources if s["source_type"] == "construction"
    )
    traffic_score = sum(
        s["contribution_score"] for s in nearby_sources if s["source_type"] == "traffic"
    )

    if construction_score == 0 and traffic_score == 0:
        dominant = "unknown"
    else:
        dominant = "construction" if construction_score > traffic_score else "traffic"

    return {
        "dominant_source": dominant,
        "construction_score": round(construction_score, 2),
        "traffic_score": round(traffic_score, 2),
    }


def get_source_attribution(station_lat, station_lon):
    """
    Convenience wrapper — call this one function from main.py.

    Returns:
        {
          "nearby_sources": [ {zone_name, source_type, distance_km, intensity, contribution_score}, ... ],
          "summary": {dominant_source, construction_score, traffic_score}
        }

    FUTURE (wind integration): once wind direction is reintroduced, multiply
    each source's contribution_score by an "upwind alignment" factor
    (0 = downwind/irrelevant, 1 = directly upwind) before summarizing —
    no other part of this module needs to change.
    """
    nearby = find_nearby_sources(station_lat, station_lon)
    summary = summarize_source_contribution(nearby)
    return {"nearby_sources": nearby, "summary": summary}