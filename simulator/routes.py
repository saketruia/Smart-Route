"""Candidate-route catalog for origin/destination pairs.

Drivers in the simulator have *habitual corridors* (home <-> work, home <-> a few places).
For every corridor we precompute several genuinely different routes (shortest, fastest at
free flow, highway-loving, local-roads-only, two random "mental map" variants).  Trips then
choose among them, so historical data contains the route diversity a future recommender
needs to learn from.

Path finding uses ``networkx`` (library Dijkstra) purely to *generate alternatives*; the
SmartRoute optimisation engine is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa

from .config import VehicleConfig
from .models import RT_ARTERIAL, RT_HIGHWAY, ZONE_DOWNTOWN, ZONE_INDUSTRIAL, ZONE_SUBURB
from .rng import make_rng
from .road_network import RoadNetwork

KIND_NAMES = ("SHORTEST", "FASTEST", "HIGHWAY_PREF", "LOCAL_PREF", "ALT_1", "ALT_2")
MAX_ROUTES_PER_OD = len(KIND_NAMES)
MAX_DISTANCE_RATIO = 1.4       # drop absurd detours relative to the shortest candidate
MAX_TIME_RATIO = 1.5
HIGHWAY_PREF_FACTORS = {RT_HIGHWAY: 0.6, RT_ARTERIAL: 0.85}   # others 1.2
LOCAL_PREF_FACTORS = {RT_HIGHWAY: 1.8, RT_ARTERIAL: 1.3}      # others 1.0
ALT_NOISE_SIGMA = 0.35
POI_ZONE_WEIGHTS = {ZONE_DOWNTOWN: 3.0, ZONE_SUBURB: 1.0, ZONE_INDUSTRIAL: 0.3}


@dataclass
class ZonePools:
    home: np.ndarray   # node indices
    work: np.ndarray
    poi: np.ndarray


@dataclass
class RouteCatalog:
    pools: ZonePools
    od_origin: np.ndarray
    od_dest: np.ndarray
    od_lookup: np.ndarray            # [n_nodes, n_nodes] -> od index or -1
    od_route_ids: np.ndarray         # [n_od, MAX_ROUTES] global route idx or -1
    od_dist_rel: np.ndarray          # route distance / shortest - 1  (padding: large)
    od_time_rel: np.ndarray          # free-flow time / fastest - 1
    od_highway: np.ndarray           # fraction of route length on highways
    od_best_ff_route: np.ndarray     # route with lowest free-flow time
    route_od: np.ndarray
    route_rank: np.ndarray
    route_kind: np.ndarray
    route_dist_km: np.ndarray
    route_ff_time_min: np.ndarray
    route_hw_frac: np.ndarray
    route_len: np.ndarray
    route_offset: np.ndarray
    flat_segments: np.ndarray
    route_node_str: list[str]
    valid_work: np.ndarray           # [H, W]
    valid_poi: np.ndarray            # [H, P]

    @property
    def n_od(self) -> int:
        return len(self.od_origin)

    @property
    def n_routes(self) -> int:
        return len(self.route_od)

    def route_ids(self) -> np.ndarray:
        return np.array([f"OD{o:05d}-R{k}" for o, k in zip(self.route_od, self.route_rank)])

    def od_ids(self) -> np.ndarray:
        return np.array([f"OD{o:05d}" for o in range(self.n_od)])

    def segments_of(self, route: int) -> np.ndarray:
        o = self.route_offset[route]
        return self.flat_segments[o:o + self.route_len[route]]

    def route_segments_arrow(self, network: RoadNetwork) -> pa.ListArray:
        """Arrow list<string> of segment ids for every route (take() it per trip)."""
        flat = pa.array(network.seg_ids[self.flat_segments])
        offsets = np.concatenate([self.route_offset, [len(self.flat_segments)]]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(offsets), flat)

    def routes_frame(self, network: RoadNetwork) -> pd.DataFrame:
        table = pa.table({
            "route_id": self.route_ids(),
            "od_id": self.od_ids()[self.route_od],
            "origin_node": network.node_ids[self.od_origin[self.route_od]],
            "destination_node": network.node_ids[self.od_dest[self.route_od]],
            "route_rank": self.route_rank,
            "route_kind": [KIND_NAMES[k] for k in self.route_kind],
            "n_segments": self.route_len.astype(np.int32),
            "distance_km": self.route_dist_km,
            "free_flow_time_min": self.route_ff_time_min,
            "highway_fraction": self.route_hw_frac,
            "route_segments": self.route_segments_arrow(network),
        })
        return table


def select_pools(network: RoadNetwork, vcfg: VehicleConfig, rng: np.random.Generator) -> ZonePools:
    """Choose home / work / leisure nodes: homes in suburbs, work downtown or industrial."""
    zone = network.node_zone
    suburb = np.flatnonzero(zone == ZONE_SUBURB)
    workish = np.flatnonzero(zone != ZONE_SUBURB)
    home = rng.choice(suburb, min(vcfg.home_zones, len(suburb)), replace=False)
    work = rng.choice(workish, min(vcfg.work_hubs, len(workish)), replace=False)
    rest = np.setdiff1d(np.arange(network.n_nodes), np.concatenate([home, work]))
    w = np.array([POI_ZONE_WEIGHTS[int(z)] for z in zone[rest]])
    poi = rng.choice(rest, min(vcfg.poi_count, len(rest)), replace=False, p=w / w.sum())
    return ZonePools(np.sort(home).astype(np.int32), np.sort(work).astype(np.int32), np.sort(poi).astype(np.int32))


def _variant_weights(net: RoadNetwork, rng: np.random.Generator) -> np.ndarray:
    ff_time = net.seg_distance_km / net.seg_baseline_speed * 60.0
    hw = np.where(net.seg_road_type == RT_HIGHWAY, HIGHWAY_PREF_FACTORS[RT_HIGHWAY],
                  np.where(net.seg_road_type == RT_ARTERIAL, HIGHWAY_PREF_FACTORS[RT_ARTERIAL], 1.2))
    local = np.where(net.seg_road_type == RT_HIGHWAY, LOCAL_PREF_FACTORS[RT_HIGHWAY],
                     np.where(net.seg_road_type == RT_ARTERIAL, LOCAL_PREF_FACTORS[RT_ARTERIAL], 1.0))
    m = net.n_segments
    return np.stack([
        net.seg_distance_km, ff_time, ff_time * hw, net.seg_distance_km * local,
        ff_time * rng.lognormal(0, ALT_NOISE_SIGMA, m), ff_time * rng.lognormal(0, ALT_NOISE_SIGMA, m),
    ])


def build_route_catalog(network: RoadNetwork, vcfg: VehicleConfig, seed: int) -> RouteCatalog:
    """Build candidate routes for all home<->work/POI corridors."""
    rng = make_rng(seed, 2)
    pools = select_pools(network, vcfg, rng)
    weights = _variant_weights(network, rng)
    ff_time = weights[1]
    graph = nx.DiGraph()
    for i in range(network.n_segments):
        graph.add_edge(int(network.seg_from[i]), int(network.seg_to[i]), idx=i,
                       **{f"w{k}": float(weights[k, i]) for k in range(MAX_ROUTES_PER_OD)})
    edge_idx = network.segment_lookup()

    away = np.concatenate([pools.work, pools.poi])
    d = np.hypot(network.node_x_km[pools.home][:, None] - network.node_x_km[away][None, :],
                 network.node_y_km[pools.home][:, None] - network.node_y_km[away][None, :])
    ok = d >= vcfg.min_trip_km                                   # [H, W+P]
    pair_dests: dict[int, np.ndarray] = {}
    for hi, h in enumerate(pools.home):
        pair_dests[int(h)] = away[ok[hi]]
    for ai, a in enumerate(away):
        pair_dests[int(a)] = pools.home[ok[:, ai]]

    od_origin, od_dest = [], []
    r_od, r_rank, r_kind, r_dist, r_ff, r_hw, r_len, r_nodes = [], [], [], [], [], [], [], []
    r_segs: list[np.ndarray] = []
    is_hw = network.seg_road_type == RT_HIGHWAY
    for src, dests in pair_dests.items():
        if len(dests) == 0:
            continue
        trees = [nx.single_source_dijkstra_path(graph, src, weight=f"w{k}") for k in range(MAX_ROUTES_PER_OD)]
        for dst in dests:
            seen: dict[tuple[int, ...], int] = {}
            for kind, tree in enumerate(trees):
                path = tuple(tree[int(dst)])
                seen.setdefault(path, kind)
            cands = []
            for path, kind in seen.items():
                segs = np.array([edge_idx[(a, b)] for a, b in zip(path, path[1:])], dtype=np.int32)
                cands.append((path, kind, segs, network.seg_distance_km[segs].sum(), ff_time[segs].sum()))
            min_d = min(c[3] for c in cands)
            min_t = min(c[4] for c in cands)
            cands = [c for c in cands if c[3] <= MAX_DISTANCE_RATIO * min_d and c[4] <= MAX_TIME_RATIO * min_t]
            if len(cands) < 2:
                continue
            od = len(od_origin)
            od_origin.append(int(src)); od_dest.append(int(dst))
            for rank, (path, kind, segs, dist, ff) in enumerate(cands):
                r_od.append(od); r_rank.append(rank); r_kind.append(kind)
                r_dist.append(dist); r_ff.append(ff)
                r_hw.append(network.seg_distance_km[segs][is_hw[segs]].sum() / dist)
                r_len.append(len(segs)); r_segs.append(segs)
                r_nodes.append(">".join(network.node_ids[list(path)]))

    n_od, n_routes = len(od_origin), len(r_od)
    route_len = np.array(r_len, dtype=np.int32)
    route_offset = np.concatenate([[0], np.cumsum(route_len)[:-1]]).astype(np.int64)
    route_od = np.array(r_od, dtype=np.int32)
    od_lookup = np.full((network.n_nodes, network.n_nodes), -1, dtype=np.int32)
    od_lookup[od_origin, od_dest] = np.arange(n_od)

    ids = np.full((n_od, MAX_ROUTES_PER_OD), -1, dtype=np.int32)
    dist_rel = np.full((n_od, MAX_ROUTES_PER_OD), 9.0, dtype=np.float32)
    time_rel = np.full((n_od, MAX_ROUTES_PER_OD), 9.0, dtype=np.float32)
    hw = np.zeros((n_od, MAX_ROUTES_PER_OD), dtype=np.float32)
    route_rank = np.array(r_rank, dtype=np.int8)
    dist_arr, ff_arr = np.array(r_dist), np.array(r_ff)
    ids[route_od, route_rank] = np.arange(n_routes)
    min_dist = np.full(n_od, np.inf); np.minimum.at(min_dist, route_od, dist_arr)
    min_ff = np.full(n_od, np.inf); np.minimum.at(min_ff, route_od, ff_arr)
    dist_rel[route_od, route_rank] = dist_arr / min_dist[route_od] - 1
    time_rel[route_od, route_rank] = ff_arr / min_ff[route_od] - 1
    hw[route_od, route_rank] = np.array(r_hw)
    best_ff = ids[np.arange(n_od), np.argmin(time_rel, axis=1)]

    def valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (od_lookup[np.ix_(a, b)] >= 0) & (od_lookup[np.ix_(b, a)].T >= 0)

    return RouteCatalog(
        pools=pools, od_origin=np.array(od_origin, dtype=np.int32), od_dest=np.array(od_dest, dtype=np.int32),
        od_lookup=od_lookup, od_route_ids=ids, od_dist_rel=dist_rel, od_time_rel=time_rel, od_highway=hw,
        od_best_ff_route=best_ff, route_od=route_od, route_rank=route_rank,
        route_kind=np.array(r_kind, dtype=np.int8), route_dist_km=dist_arr, route_ff_time_min=ff_arr,
        route_hw_frac=np.array(r_hw), route_len=route_len, route_offset=route_offset,
        flat_segments=np.concatenate(r_segs).astype(np.int32), route_node_str=r_nodes,
        valid_work=valid(pools.home, pools.work), valid_poi=valid(pools.home, pools.poi),
    )
