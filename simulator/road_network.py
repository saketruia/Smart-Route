"""Synthetic city road network.

The network is a jittered grid of intersections with:

* highway corridors every ``highway_every`` rows/columns and arterials every ``arterial_every``,
* commercial roads downtown, industrial roads in one sector, residential elsewhere,
* a few diagonal shortcuts and long "express" highway chords,
* a handful of minor roads removed (connectivity is always preserved).

Every undirected road becomes two *directed* segments, because congestion is directional
(inbound in the morning, outbound in the evening).

GEOGRAPHY IS SYNTHETIC: latitude/longitude are derived from a fictional layout around
``origin_lat/origin_lon`` purely so a dashboard can draw the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .config import NetworkConfig, TrafficConfig
from .models import (
    P_COMMERCIAL, P_EVENING, P_LOW, P_MODERATE, P_MORNING, PROFILE_NAMES, ROAD_TYPES,
    RT_ARTERIAL, RT_BASELINE_RATIO, RT_COMMERCIAL, RT_HIGHWAY, RT_INDUSTRIAL, RT_LANES_MAX,
    RT_LANES_MIN, RT_RESIDENTIAL, RT_SENSITIVITY, RT_SPEED_LIMIT, ZONE_DOWNTOWN,
    ZONE_INDUSTRIAL, ZONE_NAMES, ZONE_SUBURB,
)

KM_PER_DEG_LAT = 111.0
DOWNTOWN_RADIUS = 0.28          # fraction of the city radius
CIRCUITY_RANGE = (1.0, 1.2)     # roads are longer than the straight line
EXPRESS_CIRCUITY = 1.08
EXPRESS_SPEED_LIMIT = 100.0
EXPRESS_MIN_KM = 6.0
DIRECTION_THRESHOLD = 0.4       # cos-like ratio deciding inbound / outbound / lateral
PROBLEM_ROAD_TYPES = (RT_HIGHWAY, RT_ARTERIAL, RT_COMMERCIAL)
PROBLEM_STRENGTH_RANGE = (0.45, 0.75)
MIN_PROBLEM_SEGMENTS = 3


@dataclass
class RoadNetwork:
    """Columnar (numpy) road network. Index ``i`` in ``seg_*`` arrays is the segment index."""

    rows: int
    cols: int
    node_ids: np.ndarray
    node_x_km: np.ndarray
    node_y_km: np.ndarray
    node_lat: np.ndarray
    node_lon: np.ndarray
    node_zone: np.ndarray
    seg_ids: np.ndarray
    seg_from: np.ndarray
    seg_to: np.ndarray
    seg_distance_km: np.ndarray
    seg_speed_limit: np.ndarray
    seg_baseline_speed: np.ndarray
    seg_lanes: np.ndarray
    seg_road_type: np.ndarray
    seg_sensitivity: np.ndarray
    seg_profile: np.ndarray
    seg_heading_deg: np.ndarray
    seg_direction: np.ndarray       # +1 towards centre, -1 away from centre, 0 lateral
    seg_problem_start: np.ndarray   # hours; NaN when not a problem road
    seg_problem_end: np.ndarray
    seg_problem_strength: np.ndarray  # 0 when not a problem road

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_segments(self) -> int:
        return len(self.seg_ids)

    @property
    def problem_mask(self) -> np.ndarray:
        return self.seg_problem_strength > 0

    def segment_lookup(self) -> dict[tuple[int, int], int]:
        return {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.seg_from, self.seg_to))}

    def nodes_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "node_id": self.node_ids,
            "latitude": self.node_lat,
            "longitude": self.node_lon,
            "x_km": self.node_x_km,
            "y_km": self.node_y_km,
            "zone": [ZONE_NAMES[z] for z in self.node_zone],
        })

    def segments_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "segment_id": self.seg_ids,
            "from_node": self.node_ids[self.seg_from],
            "to_node": self.node_ids[self.seg_to],
            "distance_km": self.seg_distance_km,
            "speed_limit_kmh": self.seg_speed_limit,
            "road_type": [ROAD_TYPES[t] for t in self.seg_road_type],
            "lane_count": self.seg_lanes.astype(np.int8),
            "baseline_speed_kmh": self.seg_baseline_speed,
            "congestion_sensitivity": self.seg_sensitivity,
            "traffic_profile": [PROFILE_NAMES[p] for p in self.seg_profile],
            "heading_deg": self.seg_heading_deg,
            "from_lat": self.node_lat[self.seg_from],
            "from_lon": self.node_lon[self.seg_from],
            "to_lat": self.node_lat[self.seg_to],
            "to_lon": self.node_lon[self.seg_to],
            "is_problem_road": self.problem_mask,
            "problem_start_hour": self.seg_problem_start,
            "problem_end_hour": self.seg_problem_end,
        })

    def to_networkx(self) -> nx.DiGraph:
        """Directed graph with per-edge attributes (handy for a future routing engine)."""
        g = nx.DiGraph()
        for i in range(self.n_segments):
            g.add_edge(
                int(self.seg_from[i]), int(self.seg_to[i]), segment_index=i,
                segment_id=str(self.seg_ids[i]), distance_km=float(self.seg_distance_km[i]),
                baseline_speed_kmh=float(self.seg_baseline_speed[i]),
            )
        return g


def _plain_road_type(zone_u: int, zone_v: int) -> int:
    if ZONE_DOWNTOWN in (zone_u, zone_v):
        return RT_COMMERCIAL
    if ZONE_INDUSTRIAL in (zone_u, zone_v):
        return RT_INDUSTRIAL
    return RT_RESIDENTIAL


def _line_type(index: int, cfg: NetworkConfig) -> int | None:
    """Road type for an edge running along grid line ``index`` (row for horizontals)."""
    if index % cfg.highway_every == 0:
        return RT_HIGHWAY
    if index % cfg.arterial_every == 0:
        return RT_ARTERIAL
    return None


def _build_edges(cfg: NetworkConfig, x, y, zone, rng) -> dict[tuple[int, int], tuple[int, float]]:
    """Return ``{(u, v): (road_type, distance_km)}`` for undirected roads (u < v)."""
    rows, cols = cfg.rows, cfg.cols
    edges: dict[tuple[int, int], tuple[int, float]] = {}

    def euclid(u: int, v: int) -> float:
        return float(np.hypot(x[u] - x[v], y[u] - y[v]))

    def add(u: int, v: int, road_type: int, circuity: float | None = None) -> None:
        key = (min(u, v), max(u, v))
        if key in edges:
            return
        c = circuity if circuity is not None else rng.uniform(*CIRCUITY_RANGE)
        edges[key] = (road_type, max(euclid(u, v) * c, cfg.min_segment_km))

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:  # horizontal edge lies along row r
                add(u, u + 1, _line_type(r, cfg) or _plain_road_type(zone[u], zone[u + 1]))
            if r + 1 < rows:  # vertical edge lies along column c
                add(u, u + cols, _line_type(c, cfg) or _plain_road_type(zone[u], zone[u + cols]))
            if r + 1 < rows and c + 1 < cols and rng.random() < cfg.diagonal_prob:
                a, b = (u, u + cols + 1) if rng.random() < 0.5 else (u + 1, u + cols)
                rt = RT_ARTERIAL if rng.random() < 0.4 else _plain_road_type(zone[a], zone[b])
                add(a, b, rt)

    # Remove some minor roads while keeping the network connected.
    graph = nx.Graph(list(edges))
    minor = [k for k, (rt, _) in edges.items() if rt in (RT_RESIDENTIAL, RT_COMMERCIAL, RT_INDUSTRIAL)]
    rng.shuffle(minor)
    for key in minor:
        if rng.random() >= cfg.edge_removal_prob:
            continue
        graph.remove_edge(*key)
        if nx.is_connected(graph):
            del edges[key]
        else:
            graph.add_edge(*key)

    # Long express highway chords between corridor junctions.
    junctions = [r * cols + c for r in range(0, rows, cfg.highway_every)
                 for c in range(0, cols, cfg.highway_every)]
    added, attempts = 0, 0
    while added < cfg.express_links and len(junctions) > 1 and attempts < 200:
        attempts += 1
        u, v = (int(k) for k in rng.choice(junctions, 2, replace=False))
        same_line = (u // cols == v // cols) or (u % cols == v % cols)
        if same_line or euclid(u, v) < EXPRESS_MIN_KM or (min(u, v), max(u, v)) in edges:
            continue
        add(u, v, RT_HIGHWAY, EXPRESS_CIRCUITY)
        edges[(min(u, v), max(u, v))] = (RT_HIGHWAY, edges[(min(u, v), max(u, v))][1])
        added += 1
    return edges


def _assign_profiles(road_type, direction, rng) -> np.ndarray:
    """Recurring congestion profile per directed segment (direction-aware)."""
    m = len(road_type)
    u = rng.random(m)
    profile = np.full(m, P_LOW, dtype=np.int8)
    trunk = np.isin(road_type, (RT_ARTERIAL, RT_HIGHWAY))
    profile[trunk & (direction == 1)] = np.where(u < 0.75, P_MORNING, P_MODERATE)[trunk & (direction == 1)]
    profile[trunk & (direction == -1)] = np.where(u < 0.75, P_EVENING, P_MODERATE)[trunk & (direction == -1)]
    profile[trunk & (direction == 0)] = np.where(u < 0.6, P_MODERATE, P_LOW)[trunk & (direction == 0)]
    comm = road_type == RT_COMMERCIAL
    profile[comm] = np.where(u < 0.7, P_COMMERCIAL, P_MODERATE)[comm]
    other = ~trunk & ~comm
    profile[other] = np.where(u < 0.82, P_LOW, P_MODERATE)[other]
    return profile


def generate_road_network(cfg: NetworkConfig, traffic_cfg: TrafficConfig,
                          rng: np.random.Generator) -> RoadNetwork:
    """Build the synthetic road network. Deterministic for a given ``rng`` state."""
    rows, cols = cfg.rows, cfg.cols
    n = rows * cols
    rr, cc = np.divmod(np.arange(n), cols)
    x = (cc + rng.uniform(-cfg.jitter, cfg.jitter, n)) * cfg.spacing_km
    y = (rr + rng.uniform(-cfg.jitter, cfg.jitter, n)) * cfg.spacing_km
    cx, cy = x.mean(), y.mean()
    dist_c = np.hypot(x - cx, y - cy)
    radius = dist_c.max()

    zone = np.full(n, ZONE_SUBURB, dtype=np.int8)
    zone[((x - cx) / radius > 0.35) & ((y - cy) / radius < -0.15)] = ZONE_INDUSTRIAL
    zone[dist_c / radius < DOWNTOWN_RADIUS] = ZONE_DOWNTOWN

    edges = _build_edges(cfg, x, y, zone, rng)
    keys = sorted(edges)
    u_und = np.array([k[0] for k in keys])
    v_und = np.array([k[1] for k in keys])
    rt_und = np.array([edges[k][0] for k in keys], dtype=np.int8)
    dist_und = np.array([edges[k][1] for k in keys])

    # Two directed segments per road, ordered by (from, to).
    seg_from = np.concatenate([u_und, v_und])
    seg_to = np.concatenate([v_und, u_und])
    order = np.lexsort((seg_to, seg_from))
    seg_from, seg_to = seg_from[order].astype(np.int32), seg_to[order].astype(np.int32)
    road_type = np.concatenate([rt_und, rt_und])[order]
    distance = np.concatenate([dist_und, dist_und])[order]
    m = len(seg_from)

    limit = RT_SPEED_LIMIT[road_type].copy()
    express = (road_type == RT_HIGHWAY) & (distance / np.maximum(
        np.hypot(x[seg_from] - x[seg_to], y[seg_from] - y[seg_to]), 1e-9) < EXPRESS_CIRCUITY + 0.01) \
        & (np.hypot(x[seg_from] - x[seg_to], y[seg_from] - y[seg_to]) > EXPRESS_MIN_KM)
    limit[express] = EXPRESS_SPEED_LIMIT
    lanes = rng.integers(RT_LANES_MIN[road_type], RT_LANES_MAX[road_type] + 1)
    baseline = limit * RT_BASELINE_RATIO[road_type] * rng.uniform(0.95, 1.05, m)
    sensitivity = RT_SENSITIVITY[road_type] * (2.0 / lanes) ** 0.35 * rng.uniform(0.85, 1.15, m)

    dx, dy = x[seg_to] - x[seg_from], y[seg_to] - y[seg_from]
    length = np.hypot(dx, dy)
    heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    ratio = (dist_c[seg_to] - dist_c[seg_from]) / np.maximum(length, 1e-9)
    direction = np.where(ratio < -DIRECTION_THRESHOLD, 1, np.where(ratio > DIRECTION_THRESHOLD, -1, 0)).astype(np.int8)
    profile = _assign_profiles(road_type, direction, rng)

    # Problem roads: recurring, very high congestion inside a fixed daily window.
    p_start = np.full(m, np.nan)
    p_end = np.full(m, np.nan)
    p_strength = np.zeros(m)
    candidates = np.flatnonzero(np.isin(road_type, PROBLEM_ROAD_TYPES))
    n_problem = min(len(candidates), max(MIN_PROBLEM_SEGMENTS, round(traffic_cfg.problem_segment_fraction * m)))
    if n_problem and traffic_cfg.problem_windows:
        chosen = rng.choice(candidates, n_problem, replace=False)
        weights = np.asarray(traffic_cfg.problem_window_weights, dtype=float)
        weights = weights / weights.sum()
        win = rng.choice(len(traffic_cfg.problem_windows), n_problem, p=weights)
        windows = np.asarray(traffic_cfg.problem_windows)
        p_start[chosen], p_end[chosen] = windows[win, 0], windows[win, 1]
        p_strength[chosen] = rng.uniform(*PROBLEM_STRENGTH_RANGE, n_problem)

    lat0, lon0 = cfg.origin_lat, cfg.origin_lon
    node_lat = lat0 + (y - cy) / KM_PER_DEG_LAT
    node_lon = lon0 + (x - cx) / (KM_PER_DEG_LAT * np.cos(np.radians(lat0)))

    node_w = max(4, len(str(n - 1)))
    seg_w = max(4, len(str(m)))
    return RoadNetwork(
        rows=rows, cols=cols,
        node_ids=np.array([f"N{i:0{node_w}d}" for i in range(n)]),
        node_x_km=x, node_y_km=y, node_lat=node_lat, node_lon=node_lon, node_zone=zone,
        seg_ids=np.array([f"R{i + 1:0{seg_w}d}" for i in range(m)]),
        seg_from=seg_from, seg_to=seg_to, seg_distance_km=distance,
        seg_speed_limit=limit, seg_baseline_speed=baseline, seg_lanes=lanes.astype(np.int8),
        seg_road_type=road_type, seg_sensitivity=sensitivity, seg_profile=profile,
        seg_heading_deg=heading, seg_direction=direction,
        seg_problem_start=p_start, seg_problem_end=p_end, seg_problem_strength=p_strength,
    )
