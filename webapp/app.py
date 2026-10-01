from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SmartRoute Connected Vehicle Intelligence", version="1.1")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

vehicles_df = pd.read_parquet(DATA_DIR / "vehicles.parquet")
nodes_df = pd.read_parquet(DATA_DIR / "road_nodes.parquet")
segments_df = pd.read_parquet(DATA_DIR / "road_segments.parquet")
telemetry_df = pd.read_parquet(DATA_DIR / "telemetry.parquet")

for df, column in (
    (vehicles_df, "vehicle_id"),
    (nodes_df, "node_id"),
    (segments_df, "segment_id"),
):
    if column in df.columns:
        df[column] = df[column].astype(str)

telemetry_df["vehicle_id"] = telemetry_df["vehicle_id"].astype(str)
telemetry_df["timestamp"] = pd.to_datetime(
    telemetry_df["timestamp"], utc=True, errors="coerce"
)

route_engine = None
route_engine_error = None

try:
    from webapp.route_engine import RouteEngine
except Exception:
    try:
        from webapp.engine import RouteEngine
    except Exception:
        try:
            from route_engine import RouteEngine
        except Exception as exc:
            RouteEngine = None
            route_engine_error = exc

if RouteEngine is not None:
    try:
        route_engine = RouteEngine(data_dir=DATA_DIR)
    except TypeError:
        route_engine = RouteEngine(DATA_DIR)
    except Exception as exc:
        route_engine_error = exc
        route_engine = None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json_safe(df.to_dict(orient="records"))


def get_vehicle(vehicle_id: str) -> pd.Series:
    rows = vehicles_df[
        vehicles_df["vehicle_id"].astype(str).eq(str(vehicle_id))
    ]
    if rows.empty:
        raise HTTPException(404, "Vehicle not found.")
    return rows.iloc[0]


def get_latest_telemetry(vehicle_id: str) -> Optional[dict[str, Any]]:
    rows = telemetry_df[
        telemetry_df["vehicle_id"].astype(str).eq(str(vehicle_id))
    ].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("timestamp")
    return json_safe(rows.iloc[-1].to_dict())


def nearest_node(latitude: float, longitude: float) -> str:
    lat = pd.to_numeric(nodes_df["latitude"], errors="coerce")
    lon = pd.to_numeric(nodes_df["longitude"], errors="coerce")
    distance = (lat - float(latitude)) ** 2 + (lon - float(longitude)) ** 2
    return str(nodes_df.loc[distance.idxmin(), "node_id"])


def vehicle_current_node(vehicle_id: str) -> Optional[str]:
    latest = get_latest_telemetry(vehicle_id)
    if not latest:
        return None
    if latest.get("latitude") is None or latest.get("longitude") is None:
        return None
    return nearest_node(
        float(latest["latitude"]),
        float(latest["longitude"]),
    )


def map_priority(value: Optional[str]) -> str:
    aliases = {
        "balanced": "balanced",
        "smart": "balanced",
        "fast": "time",
        "fastest": "time",
        "time": "time",
        "fuel": "fuel",
        "eco": "fuel",
        "efficient": "fuel",
        "short": "distance",
        "shortest": "distance",
        "distance": "distance",
    }
    return aliases.get(str(value or "balanced").lower(), "balanced")


def build_destinations(source_node: Optional[str] = None) -> list[dict[str, str]]:
    """
    Pick eight deterministic destinations spread across the real road network.
    They are labels only; routing still uses the actual node IDs and real graph.
    """
    work = nodes_df[["node_id", "latitude", "longitude"]].copy()
    work["node_id"] = work["node_id"].astype(str)

    if source_node:
        source_rows = work[work["node_id"].eq(str(source_node))]
        if not source_rows.empty:
            slat = float(source_rows.iloc[0]["latitude"])
            slon = float(source_rows.iloc[0]["longitude"])
            work["distance_from_source"] = (
                (work["latitude"] - slat) ** 2
                + (work["longitude"] - slon) ** 2
            )
            work = work.sort_values("distance_from_source", ascending=False)

    # Deterministic spread through the network rather than four hard-coded options.
    candidate_count = min(8, len(work))
    positions = np.linspace(0, len(work) - 1, candidate_count, dtype=int)
    chosen = work.iloc[positions].drop_duplicates("node_id")

    names = [
        "North District",
        "Airport Corridor",
        "Tech Park",
        "University",
        "Central Market",
        "Riverside",
        "Stadium",
        "Industrial Hub",
    ]

    result = []
    for index, (_, row) in enumerate(chosen.iterrows()):
        node_id = str(row["node_id"])
        if source_node and node_id == str(source_node):
            continue
        result.append(
            {
                "id": node_id,
                "name": names[len(result) % len(names)],
                "label": f"{names[len(result) % len(names)]} · {node_id}",
            }
        )
    return result


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "route_engine": route_engine is not None,
        "route_engine_error": str(route_engine_error) if route_engine_error else None,
    }


@app.get("/api/summary")
def summary():
    return {
        "vehicles": int(len(vehicles_df)),
        "nodes": int(len(nodes_df)),
        "segments": int(len(segments_df)),
        "telemetry": int(len(telemetry_df)),
        "telemetry_events": int(len(telemetry_df)),
        "route_engine": route_engine is not None,
    }


@app.get("/api/network")
def network():
    if route_engine is not None:
        return json_safe(route_engine.network_payload())

    return {
        "nodes": [
            {
                "id": str(row.node_id),
                "lat": float(row.latitude),
                "lon": float(row.longitude),
            }
            for row in nodes_df.itertuples(index=False)
        ],
        "segments": [
            {
                "id": str(row.segment_id),
                "from": str(row.from_node),
                "to": str(row.to_node),
            }
            for row in segments_df.itertuples(index=False)
        ],
    }


@app.get("/api/nodes")
def nodes(limit: int = 150):
    limit = max(1, min(int(limit), len(nodes_df)))

    if route_engine is not None and hasattr(route_engine, "node_choices"):
        return json_safe(route_engine.node_choices(limit=limit))

    indices = (
        np.arange(len(nodes_df))
        if len(nodes_df) <= limit
        else np.linspace(0, len(nodes_df) - 1, limit, dtype=int)
    )

    return [
        {
            "id": str(nodes_df.iloc[int(i)]["node_id"]),
            "label": str(nodes_df.iloc[int(i)]["node_id"]),
        }
        for i in indices
    ]


@app.get("/api/destinations")
def destinations(vehicle_id: Optional[str] = None):
    source = vehicle_current_node(vehicle_id) if vehicle_id else None
    return build_destinations(source)


@app.get("/api/vehicles")
def vehicles():
    return records(vehicles_df.head(80))


@app.get("/api/vehicle/{vehicle_id}")
def vehicle(vehicle_id: str):
    row = get_vehicle(vehicle_id)
    latest = get_latest_telemetry(vehicle_id)

    current_node = None
    location = None

    if latest:
        if latest.get("latitude") is not None and latest.get("longitude") is not None:
            location = {
                "lat": float(latest["latitude"]),
                "lon": float(latest["longitude"]),
            }
            current_node = nearest_node(
                location["lat"],
                location["lon"],
            )

    return {
        "profile": json_safe(row.to_dict()),
        "current_node": current_node,
        "location": location,
        "live": latest,
        "latest_telemetry": latest,
    }


@app.get("/api/telemetry/{vehicle_id}")
def vehicle_telemetry(vehicle_id: str, limit: int = 50):
    get_vehicle(vehicle_id)
    rows = telemetry_df[
        telemetry_df["vehicle_id"].astype(str).eq(str(vehicle_id))
    ].sort_values("timestamp").tail(max(1, min(int(limit), 500)))
    return records(rows)


@app.get("/api/analytics")
def analytics():
    t = telemetry_df.copy()
    t["hour"] = t["timestamp"].dt.hour

    hourly = (
        t.groupby("hour")["traffic_level"]
        .mean()
        .reindex(range(24), fill_value=0)
        .round(4)
        .tolist()
    )

    fleet_mix = {}
    if "vehicle_type" in vehicles_df.columns:
        fleet_mix = (
            vehicles_df["vehicle_type"]
            .astype(str)
            .value_counts()
            .head(8)
            .to_dict()
        )

    return {
        "hourly_traffic": hourly,
        "fleet_mix": fleet_mix,
        "events": int(len(t)),
        "vehicles": int(len(vehicles_df)),
    }


@app.api_route("/api/route", methods=["GET", "POST"])
def calculate_route(
    source: Optional[str] = None,
    destination: Optional[str] = None,
    priority: str = "balanced",
    hour: int = 18,
    vehicle_id: Optional[str] = None,
    mode: Optional[str] = None,
):
    if route_engine is None:
        detail = "RouteEngine could not be loaded."
        if route_engine_error:
            detail += f" {route_engine_error}"
        raise HTTPException(500, detail)

    priority = map_priority(mode if mode is not None else priority)

    if vehicle_id and not source:
        source = vehicle_current_node(vehicle_id)

    if not source:
        raise HTTPException(400, "Missing source/current vehicle position.")
    if not destination:
        raise HTTPException(400, "Missing destination.")

    try:
        hour = max(0, min(23, int(hour)))
    except Exception:
        hour = 18

    try:
        return json_safe(
            route_engine.optimize(
                source=str(source),
                destination=str(destination),
                priority=priority,
                hour=hour,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        print("Route calculation error:", repr(exc))
        raise HTTPException(500, f"Route calculation failed: {exc}")


@app.on_event("startup")
def startup_message():
    print()
    print("=" * 60)
    print("SmartRoute Connected Vehicle Intelligence")
    print("=" * 60)
    print(f"Data directory : {DATA_DIR}")
    print(f"Vehicles       : {len(vehicles_df):,}")
    print(f"Road nodes     : {len(nodes_df):,}")
    print(f"Road segments  : {len(segments_df):,}")
    print(f"Telemetry rows : {len(telemetry_df):,}")
    print("Route engine   : " + ("READY" if route_engine else "NOT LOADED"))
    print("=" * 60)
    print()
