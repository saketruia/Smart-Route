from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .engine import RouteEngine


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="SmartRoute Connected Vehicle Intelligence")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

engine: RouteEngine | None = None
vehicles: pd.DataFrame | None = None
telemetry: pd.DataFrame | None = None
trips: pd.DataFrame | None = None


def _clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    return v


def _nearest_node(lat: float, lon: float) -> str:
    n = engine.nodes
    d = (n["latitude"] - lat) ** 2 + (n["longitude"] - lon) ** 2
    return str(n.iloc[int(d.argmin())]["node_id"])


def _vehicle_row(vehicle_id: str):
    matches = vehicles[vehicles["vehicle_id"].astype(str) == str(vehicle_id)]
    if matches.empty:
        raise HTTPException(404, "Vehicle not found")
    return matches.iloc[0]


@app.on_event("startup")
def startup():
    global engine, vehicles, telemetry, trips
    engine = RouteEngine(DATA_DIR)
    vehicles = pd.read_parquet(DATA_DIR / "vehicles.parquet")
    telemetry = pd.read_parquet(DATA_DIR / "telemetry.parquet")
    trips = pd.read_parquet(DATA_DIR / "trips.parquet")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/summary")
def summary():
    return engine.summary()


@app.get("/api/network")
def network():
    return engine.network_payload()


@app.get("/api/nodes")
def nodes():
    return engine.node_choices(220)


@app.get("/api/vehicles")
def vehicle_list():
    cols = [c for c in [
        "vehicle_id", "vehicle_type", "fuel_type",
        "rated_efficiency_kmpl", "driving_style", "wear_factor"
    ] if c in vehicles.columns]
    rows = []
    for _, r in vehicles.head(80).iterrows():
        rows.append({c: _clean_value(r[c]) for c in cols})
    return rows


@app.get("/api/vehicle/{vehicle_id}")
def vehicle(vehicle_id: str):
    row = _vehicle_row(vehicle_id)

    tv = telemetry[telemetry["vehicle_id"].astype(str) == str(vehicle_id)].copy()
    if tv.empty:
        raise HTTPException(404, "No telemetry for this vehicle")

    tv["timestamp"] = pd.to_datetime(tv["timestamp"], utc=True, errors="coerce")
    tv = tv.sort_values("timestamp")
    latest = tv.iloc[-1]

    lat = float(latest["latitude"])
    lon = float(latest["longitude"])
    current_node = _nearest_node(lat, lon)

    profile = {c: _clean_value(row[c]) for c in vehicles.columns}
    live = {c: _clean_value(latest[c]) for c in [
        "timestamp", "segment_id", "trip_id", "latitude", "longitude",
        "speed_kmh", "traffic_level", "fuel_level_pct", "vehicle_state",
        "event_kind", "distance_km", "fuel_consumed_l"
    ] if c in latest.index}

    return {
        "profile": profile,
        "current_node": current_node,
        "location": {"lat": lat, "lon": lon},
        "live": live,
    }


@app.get("/api/telemetry/{vehicle_id}")
def vehicle_telemetry(vehicle_id: str, limit: int = Query(24, ge=5, le=120)):
    tv = telemetry[telemetry["vehicle_id"].astype(str) == str(vehicle_id)].copy()
    if tv.empty:
        raise HTTPException(404, "No telemetry for this vehicle")
    tv["timestamp"] = pd.to_datetime(tv["timestamp"], utc=True, errors="coerce")
    tv = tv.sort_values("timestamp").tail(limit)

    out = []
    for _, r in tv.iterrows():
        out.append({
            "timestamp": r["timestamp"].isoformat() if pd.notna(r["timestamp"]) else None,
            "speed_kmh": _clean_value(r.get("speed_kmh")),
            "traffic_level": _clean_value(r.get("traffic_level")),
            "fuel_level_pct": _clean_value(r.get("fuel_level_pct")),
            "fuel_consumed_l": _clean_value(r.get("fuel_consumed_l")),
            "segment_id": _clean_value(r.get("segment_id")),
            "latitude": _clean_value(r.get("latitude")),
            "longitude": _clean_value(r.get("longitude")),
        })
    return out


@app.get("/api/analytics")
def analytics():
    t = telemetry.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
    t["hour"] = t["timestamp"].dt.hour

    hourly = (
        t.groupby("hour", observed=True)["traffic_level"]
        .mean()
        .reindex(range(24), fill_value=0)
        .round(4)
        .tolist()
    )

    fleet = {}
    if "vehicle_type" in vehicles.columns:
        fleet = vehicles["vehicle_type"].astype(str).value_counts().head(8).to_dict()

    fuel_style = {}
    if "vehicle_id" in t.columns and "driving_style" in vehicles.columns:
        join = t[["vehicle_id", "fuel_consumed_l", "distance_km"]].copy()
        meta = vehicles[["vehicle_id", "driving_style"]].copy()
        join["vehicle_id"] = join["vehicle_id"].astype(str)
        meta["vehicle_id"] = meta["vehicle_id"].astype(str)
        j = join.merge(meta, on="vehicle_id", how="left")
        j["fuel_per_km"] = j["fuel_consumed_l"] / j["distance_km"].replace(0, np.nan)
        fuel_style = (
            j.groupby("driving_style", observed=True)["fuel_per_km"]
            .mean()
            .dropna()
            .round(5)
            .to_dict()
        )

    return {
        "hourly_traffic": hourly,
        "fleet_mix": fleet,
        "fuel_per_km_by_style": fuel_style,
        "events": int(len(t)),
        "vehicles": int(len(vehicles)),
        "trips": int(len(trips)),
    }


@app.post("/api/route")
def route(payload: dict):
    try:
        return engine.optimize(
            payload.get("source"),
            payload.get("destination"),
            payload.get("priority", "balanced"),
            int(payload.get("hour", 18)),
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))
