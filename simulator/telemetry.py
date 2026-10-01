"""Telemetry engine: turns planned trips into per-segment observations, trips and Arrow tables.

Trips are simulated *in lock-step*: iteration ``j`` advances the j-th segment of every trip
that still has one, so a 20K-trip chunk needs only ``max_route_len`` vectorised passes.
Each segment's traffic is evaluated at the trip's *own* clock, so a long trip that starts
in the evening rush can leave it behind - exactly the time-dependence a route optimiser
has to learn.

One telemetry event is emitted per segment traversal (at segment exit, ``SEGMENT_COMPLETE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from . import dynamics as dyn
from .models import EVENT_KINDS, EVENT_SEGMENT_COMPLETE, TRIP_KINDS, VEHICLE_STATES
from .rng import format_event_ids, hash_keys
from .routes import KIND_NAMES
from .trips import TripPlan
from .world import World

FUEL_START_PCT_RANGE = (25.0, 95.0)
FUEL_RESERVE_PCT = 5.0
LATENCY_MEDIAN_MS, LATENCY_SIGMA = 250.0, 0.6
_S_EVENT_ID = 11

_DICT = pa.dictionary(pa.int32(), pa.string())
_TS = pa.timestamp("ms", tz="UTC")
TELEMETRY_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("vehicle_id", _DICT), ("timestamp", _TS), ("emitted_at", _TS),
    ("trip_id", _DICT), ("segment_id", _DICT), ("segment_seq", pa.int16()),
    ("latitude", pa.float64()), ("longitude", pa.float64()), ("heading_deg", pa.float32()),
    ("speed_kmh", pa.float32()), ("traffic_level", pa.float32()), ("fuel_consumed_l", pa.float32()),
    ("distance_km", pa.float32()), ("travel_time_min", pa.float32()), ("fuel_level_pct", pa.float32()),
    ("vehicle_state", _DICT), ("event_kind", _DICT),
])
TRIPS_SCHEMA = pa.schema([
    ("trip_id", pa.string()), ("vehicle_id", _DICT), ("origin_node", _DICT), ("destination_node", _DICT),
    ("trip_kind", _DICT), ("od_id", pa.string()), ("route_id", pa.string()), ("route_kind", _DICT),
    ("route", pa.string()), ("start_timestamp", _TS), ("end_timestamp", _TS),
    ("total_distance_km", pa.float64()), ("total_time_min", pa.float64()), ("total_fuel_l", pa.float64()),
    ("average_speed_kmh", pa.float64()), ("n_segments", pa.int16()),
    ("route_segments", pa.list_(pa.string())),
])


@dataclass
class TraversalBatch:
    """Flat per-segment arrays (length = total traversals) plus per-trip aggregates."""
    trip_of: np.ndarray
    seq: np.ndarray
    seg: np.ndarray
    t_enter: np.ndarray
    t_exit: np.ndarray
    speed: np.ndarray
    level: np.ndarray
    dist: np.ndarray
    tt_min: np.ndarray
    fuel: np.ndarray
    fuel_level_pct: np.ndarray
    trip_off: np.ndarray
    route_len: np.ndarray
    trip_dist: np.ndarray
    trip_time_min: np.ndarray
    trip_fuel: np.ndarray
    trip_end_s: np.ndarray


def simulate_traversals(world: World, plan: TripPlan, rng: np.random.Generator) -> TraversalBatch:
    net, cat, fleet, tr, p = world.network, world.catalog, world.fleet, world.traffic, world.dynamics
    n_trips = len(plan)
    route_len = cat.route_len[plan.route].astype(np.int64)
    trip_off = np.concatenate([[0], np.cumsum(route_len)[:-1]])
    total = int(route_len.sum())
    trip_of = np.repeat(np.arange(n_trips), route_len)
    seq = np.arange(total) - trip_off[trip_of]
    seg_flat = cat.flat_segments[cat.route_offset[plan.route][trip_of] + seq]

    veh = plan.vehicle_idx
    trip_speed_mult = np.exp(rng.normal(0.0, p.trip_speed_sigma, n_trips))
    trip_fuel_mult = np.exp(rng.normal(0.0, p.trip_fuel_sigma, n_trips)) * fleet.wear[veh]
    speed_z = rng.standard_normal(total)
    fuel_z = rng.standard_normal(total)

    clock = plan.start_s.astype(np.float64).copy()
    t_enter = np.empty(total); speed = np.empty(total); level = np.empty(total)
    for j in range(int(route_len.max())):
        active = np.flatnonzero(route_len > j)
        pos = trip_off[active] + j
        seg = seg_flat[pos]
        lvl = tr.level(seg, clock[active])
        v_idx = veh[active]
        spd = dyn.effective_speed(net.seg_baseline_speed[seg], net.seg_speed_limit[seg], lvl,
                                  fleet.speed_factor[v_idx], fleet.driving_style[v_idx],
                                  trip_speed_mult[active], speed_z[pos], p)
        t_enter[pos], speed[pos], level[pos] = clock[active], spd, lvl
        clock[active] += net.seg_distance_km[seg] / spd * 3600.0

    dist = net.seg_distance_km[seg_flat]
    tt_min = dyn.travel_time_min(dist, speed)
    v_of = veh[trip_of]
    fuel = dyn.fuel_litres(dist, fleet.kmpl[v_of], speed, level, fleet.driving_style[v_of],
                           fleet.fuel_type[v_of], trip_fuel_mult[trip_of], fuel_z, p)
    t_exit = t_enter + tt_min * 60.0

    trip_dist = np.add.reduceat(dist, trip_off)
    trip_time = np.add.reduceat(tt_min, trip_off)
    trip_fuel = np.add.reduceat(fuel, trip_off)

    tank = fleet.tank_l[veh]
    need_pct = trip_fuel / tank * 100.0
    start_pct = np.maximum(rng.uniform(*FUEL_START_PCT_RANGE, n_trips), need_pct + FUEL_RESERVE_PCT)
    start_pct = np.minimum(start_pct, 100.0)
    cum = np.cumsum(fuel)
    before = np.concatenate([[0.0], cum])[trip_off]
    fuel_level = np.clip(start_pct[trip_of] - (cum - before[trip_of]) / tank[trip_of] * 100.0, 0.0, 100.0)

    return TraversalBatch(trip_of, seq, seg_flat, t_enter, t_exit, speed, level, dist, tt_min, fuel,
                          fuel_level, trip_off, route_len, trip_dist, trip_time, trip_fuel, clock)


def _dict_array(indices: np.ndarray, values: pa.Array) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(indices.astype(np.int32)), values)


def _ts_ms(epoch_s: np.ndarray) -> pa.Array:
    return pa.array(np.rint(epoch_s * 1000.0).astype(np.int64), type=_TS)


class TableBuilder:
    """Builds Arrow tables for a chunk; holds Arrow copies of static lookups."""

    def __init__(self, world: World):
        self.w = world
        net, cat = world.network, world.catalog
        self.seg_ids = pa.array(net.seg_ids)
        self.node_ids = pa.array(net.node_ids)
        self.state_vals = pa.array(VEHICLE_STATES)
        self.kind_vals = pa.array(EVENT_KINDS)
        self.trip_kind_vals = pa.array(TRIP_KINDS)
        self.route_kind_vals = pa.array(KIND_NAMES)
        self.route_ids = pa.array(cat.route_ids())
        self.od_ids = pa.array(cat.od_ids())
        self.route_nodes = pa.array(cat.route_node_str)
        self.route_segs = cat.route_segments_arrow(net)
        self.trip_lat = net.node_lat[net.seg_to]
        self.trip_lon = net.node_lon[net.seg_to]

    @staticmethod
    def trip_ids(trip_no: np.ndarray) -> np.ndarray:
        return np.array([f"T{n:08d}" for n in trip_no])

    def telemetry(self, plan: TripPlan, tb: TraversalBatch, v_start: int, v_stop: int,
                  rng: np.random.Generator) -> pa.Table:
        w, net = self.w, self.w.network
        order = np.argsort(tb.t_exit, kind="stable")
        trip_of, seq, seg = tb.trip_of[order], tb.seq[order], tb.seg[order]
        t_exit, speed = tb.t_exit[order], tb.speed[order]
        ids = format_event_ids(hash_keys(w.cfg.seed, _S_EVENT_ID, plan.trip_no[trip_of], seq))
        ts_ms = np.rint(t_exit * 1000.0).astype(np.int64)
        latency = np.exp(rng.normal(np.log(LATENCY_MEDIAN_MS), LATENCY_SIGMA, len(order))).astype(np.int64)
        state = dyn.vehicle_state_code(speed, net.seg_baseline_speed[seg])
        veh_dict = pa.array(w.fleet.ids(v_start, v_stop))
        trip_dict = pa.array(self.trip_ids(plan.trip_no))
        arrays = [
            pa.array(ids).cast(pa.string()),
            _dict_array(plan.vehicle_idx[trip_of] - v_start, veh_dict),
            pa.array(ts_ms, type=_TS), pa.array(ts_ms + latency, type=_TS),
            _dict_array(trip_of, trip_dict), _dict_array(seg, self.seg_ids),
            pa.array(seq.astype(np.int16)),
            pa.array(self.trip_lat[seg]), pa.array(self.trip_lon[seg]),
            pa.array(net.seg_heading_deg[seg].astype(np.float32)),
            pa.array(speed.astype(np.float32)), pa.array(tb.level[order].astype(np.float32)),
            pa.array(tb.fuel[order].astype(np.float32)), pa.array(tb.dist[order].astype(np.float32)),
            pa.array(tb.tt_min[order].astype(np.float32)), pa.array(tb.fuel_level_pct[order].astype(np.float32)),
            _dict_array(state, self.state_vals),
            _dict_array(np.full(len(order), EVENT_SEGMENT_COMPLETE), self.kind_vals),
        ]
        return pa.Table.from_arrays(arrays, schema=TELEMETRY_SCHEMA)

    def trips(self, plan: TripPlan, tb: TraversalBatch, v_start: int, v_stop: int) -> pa.Table:
        w, cat = self.w, self.w.catalog
        route_idx = pa.array(plan.route.astype(np.int64))
        veh_dict = pa.array(w.fleet.ids(v_start, v_stop))
        avg_speed = tb.trip_dist / (tb.trip_time_min / 60.0)
        arrays = [
            pa.array(self.trip_ids(plan.trip_no)),
            _dict_array(plan.vehicle_idx - v_start, veh_dict),
            _dict_array(plan.origin, self.node_ids), _dict_array(plan.dest, self.node_ids),
            _dict_array(plan.kind, self.trip_kind_vals),
            self.od_ids.take(pa.array(plan.od.astype(np.int64))),
            self.route_ids.take(route_idx),
            _dict_array(cat.route_kind[plan.route], self.route_kind_vals),
            self.route_nodes.take(route_idx),
            _ts_ms(plan.start_s), _ts_ms(tb.trip_end_s),
            pa.array(tb.trip_dist), pa.array(tb.trip_time_min), pa.array(tb.trip_fuel), pa.array(avg_speed),
            pa.array(tb.route_len.astype(np.int16)),
            self.route_segs.take(route_idx),
        ]
        return pa.Table.from_arrays(arrays, schema=TRIPS_SCHEMA)
