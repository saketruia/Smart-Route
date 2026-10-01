from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class PathResult:
    nodes: list[str]
    segments: list[str]
    distance_km: float
    time_min: float
    fuel_l: float
    traffic: float
    score: float


class RouteEngine:
    """Historical-data route optimizer."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self._load()

    def _load(self) -> None:
        nodes_path = self.data_dir / "road_nodes.parquet"
        seg_path = self.data_dir / "road_segments.parquet"
        tel_path = self.data_dir / "telemetry.parquet"

        missing = [str(p) for p in (nodes_path, seg_path, tel_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "SmartRoute data is missing. Missing: " + ", ".join(missing)
            )

        self.nodes = pd.read_parquet(nodes_path)
        self.segments = pd.read_parquet(seg_path)
        self.telemetry = pd.read_parquet(tel_path)

        self.node_lookup = self.nodes.set_index("node_id").to_dict("index")

        t = self.telemetry.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
        t["hour"] = t["timestamp"].dt.hour

        if "event_kind" in t.columns:
            hist = t[t["event_kind"].astype(str).eq("SEGMENT_COMPLETE")].copy()
            if hist.empty:
                hist = t.copy()
        else:
            hist = t

        hist["distance_km"] = pd.to_numeric(hist["distance_km"], errors="coerce").clip(lower=0)
        hist["travel_time_min"] = pd.to_numeric(hist["travel_time_min"], errors="coerce").clip(lower=0)
        hist["fuel_consumed_l"] = pd.to_numeric(hist["fuel_consumed_l"], errors="coerce").clip(lower=0)
        hist["traffic_level"] = pd.to_numeric(hist["traffic_level"], errors="coerce").clip(0, 1)
        self.hist = hist

        self.overall = (
            hist.groupby("segment_id", as_index=False, observed=True)
            .agg(
                traffic_level=("traffic_level", "mean"),
                travel_time_min=("travel_time_min", "mean"),
                fuel_consumed_l=("fuel_consumed_l", "mean"),
                observations=("event_id", "count"),
            )
        )

        self.hourly = (
            hist.groupby(["hour", "segment_id"], as_index=False, observed=True)
            .agg(
                traffic_level=("traffic_level", "mean"),
                travel_time_min=("travel_time_min", "mean"),
                fuel_consumed_l=("fuel_consumed_l", "mean"),
                observations=("event_id", "count"),
            )
        )

        self.edge_base = self.segments[
            ["segment_id", "from_node", "to_node", "distance_km",
             "speed_limit_kmh", "road_type"]
        ].copy()

        self.available_nodes = set(self.nodes["node_id"].astype(str))
        self._network_cache: dict[int, tuple[nx.DiGraph, pd.DataFrame]] = {}

    @staticmethod
    def _norm(series: pd.Series) -> pd.Series:
        lo, hi = float(series.min()), float(series.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - lo) / (hi - lo)

    def _edge_metrics(self, hour: int) -> pd.DataFrame:
        hour = int(hour) if int(hour) in range(24) else 18
        h = self.hourly[self.hourly["hour"] == hour]

        m = self.edge_base.merge(
            h.drop(columns=["hour"]), on="segment_id", how="left"
        )
        m = m.merge(
            self.overall.rename(columns={
                "traffic_level": "traffic_overall",
                "travel_time_min": "travel_overall",
                "fuel_consumed_l": "fuel_overall",
                "observations": "obs_overall",
            }),
            on="segment_id", how="left"
        )

        m["traffic_level"] = m["traffic_level"].fillna(m["traffic_overall"]).fillna(0.3)
        m["travel_time_min"] = m["travel_time_min"].fillna(m["travel_overall"])
        m["fuel_consumed_l"] = m["fuel_consumed_l"].fillna(m["fuel_overall"])

        m["travel_time_min"] = m["travel_time_min"].fillna(
            m["distance_km"] / m["speed_limit_kmh"].clip(lower=1) * 60.0
        )
        m["fuel_consumed_l"] = m["fuel_consumed_l"].fillna(m["distance_km"] / 14.0)

        m["traffic_level"] = m["traffic_level"].clip(0, 1)
        m["travel_time_min"] = m["travel_time_min"].clip(lower=0.001)
        m["fuel_consumed_l"] = m["fuel_consumed_l"].clip(lower=0)

        m["n_time"] = self._norm(m["travel_time_min"])
        m["n_fuel"] = self._norm(m["fuel_consumed_l"])
        m["n_distance"] = self._norm(m["distance_km"])
        m["n_traffic"] = self._norm(m["traffic_level"])
        return m

    @staticmethod
    def _weights(priority: str) -> dict[str, float]:
        return {
            "balanced": {"time": .35, "fuel": .30, "traffic": .20, "distance": .15},
            "time": {"time": .60, "fuel": .10, "traffic": .20, "distance": .10},
            "fuel": {"time": .15, "fuel": .60, "traffic": .15, "distance": .10},
            "distance": {"time": .15, "fuel": .10, "traffic": .10, "distance": .65},
        }.get(priority, {"time": .35, "fuel": .30, "traffic": .20, "distance": .15})

    def _build_graph(self, hour: int):
        key = int(hour)
        if key in self._network_cache:
            return self._network_cache[key]

        metrics = self._edge_metrics(key)
        g = nx.DiGraph()

        for row in metrics.itertuples(index=False):
            g.add_edge(
                str(row.from_node), str(row.to_node),
                segment_id=str(row.segment_id),
                distance_km=float(row.distance_km),
                time_min=float(row.travel_time_min),
                fuel_l=float(row.fuel_consumed_l),
                traffic=float(row.traffic_level),
                n_time=float(row.n_time),
                n_fuel=float(row.n_fuel),
                n_distance=float(row.n_distance),
                n_traffic=float(row.n_traffic),
            )

        self._network_cache[key] = (g, metrics)
        return g, metrics

    def _path_result(self, g, path, score=0.0):
        segments, dist, tm, fuel, traffic = [], 0.0, 0.0, 0.0, 0.0
        for u, v in zip(path[:-1], path[1:]):
            d = g[u][v]
            segments.append(d["segment_id"])
            dist += d["distance_km"]
            tm += d["time_min"]
            fuel += d["fuel_l"]
            traffic += d["traffic"]
        avg_traffic = traffic / max(len(segments), 1)
        return PathResult(path, segments, dist, tm, fuel, avg_traffic, score)

    def _score_weight(self, data, w):
        return (
            w["time"] * data["n_time"]
            + w["fuel"] * data["n_fuel"]
            + w["traffic"] * data["n_traffic"]
            + w["distance"] * data["n_distance"]
        )

    def optimize(self, source, destination, priority="balanced", hour=18):
        source, destination = str(source), str(destination)
        if source not in self.available_nodes or destination not in self.available_nodes:
            raise ValueError("Source and destination must be valid road node IDs.")
        if source == destination:
            raise ValueError("Source and destination must be different.")

        g, _ = self._build_graph(hour)
        if not nx.has_path(g, source, destination):
            raise ValueError("No route exists between the selected nodes.")

        w = self._weights(priority)

        def weighted(u, v, d):
            return self._score_weight(d, w) + 1e-9

        recommended_path = nx.shortest_path(g, source, destination, weight=weighted)
        recommended = self._path_result(g, recommended_path)

        shortest = nx.shortest_path(g, source, destination, weight="distance_km")
        fastest = nx.shortest_path(g, source, destination, weight="time_min")
        fuel = nx.shortest_path(g, source, destination, weight="fuel_l")

        candidates = [
            ("Shortest distance", shortest),
            ("Fastest historical time", fastest),
            ("Lowest historical fuel", fuel),
        ]

        results = []
        seen = {tuple(recommended_path)}
        for label, path in candidates:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            r = self._path_result(g, path)
            results.append({"label": label, **self._serialize(r)})

        score = sum(
            weighted(u, v, g[u][v])
            for u, v in zip(recommended_path[:-1], recommended_path[1:])
        )
        recommended.score = score

        reasons = []
        if priority == "fuel":
            reasons.append("Fuel efficiency is weighted highest for this request.")
        elif priority == "time":
            reasons.append("Travel time is weighted highest for this request.")
        elif priority == "distance":
            reasons.append("Distance is weighted highest for this request.")
        else:
            reasons.append("The route balances historical time, fuel, congestion and distance.")

        if recommended.traffic < 0.45:
            reasons.append("The selected route has relatively low historical congestion.")
        else:
            reasons.append("The route accepts some congestion to avoid larger time/fuel costs elsewhere.")

        if results and recommended.distance_km > results[0]["distance_km"] * 1.03:
            reasons.append("The recommendation may be longer than the shortest route because route cost is multi-objective.")

        return {
            "source": source,
            "destination": destination,
            "priority": priority,
            "hour": int(hour),
            "recommended": {"label": "SmartRoute recommendation", **self._serialize(recommended)},
            "alternatives": results,
            "explanation": reasons,
        }

    @staticmethod
    def _serialize(r):
        return {
            "nodes": r.nodes,
            "segments": r.segments,
            "distance_km": round(r.distance_km, 2),
            "time_min": round(r.time_min, 2),
            "fuel_l": round(r.fuel_l, 3),
            "traffic_pct": round(r.traffic * 100, 1),
            "score": round(r.score, 4),
        }

    def network_payload(self):
        nodes = [
            {"id": str(r.node_id), "lat": float(r.latitude), "lon": float(r.longitude)}
            for r in self.nodes.itertuples(index=False)
        ]
        segments = [
            {"id": str(r.segment_id), "from": str(r.from_node), "to": str(r.to_node)}
            for r in self.segments.itertuples(index=False)
        ]
        return {"nodes": nodes, "segments": segments}

    def summary(self):
        vehicles = pd.read_parquet(self.data_dir / "vehicles.parquet")
        trips = pd.read_parquet(self.data_dir / "trips.parquet")
        return {
            "vehicles": int(len(vehicles)),
            "trips": int(len(trips)),
            "telemetry_events": int(len(self.telemetry)),
            "road_segments": int(len(self.segments)),
            "road_nodes": int(len(self.nodes)),
            "avg_traffic_pct": round(float(self.telemetry["traffic_level"].mean() * 100), 1),
            "avg_trip_distance_km": round(float(trips["total_distance_km"].mean()), 1),
            "avg_trip_time_min": round(float(trips["total_time_min"].mean()), 1),
            "avg_trip_fuel_l": round(float(trips["total_fuel_l"].mean()), 2),
        }

    def node_choices(self, limit=150):
        n = len(self.nodes)
        idx = np.linspace(0, n - 1, min(limit, n), dtype=int)
        return [
            {"id": str(self.nodes.iloc[i]["node_id"]), "label": str(self.nodes.iloc[i]["node_id"])}
            for i in idx
        ]
