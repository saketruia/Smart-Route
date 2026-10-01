"""Live mode: a stateful fleet that emits telemetry as simulated time advances.

Each vehicle cycles through home -> work/leisure -> home, drives its chosen route segment by
segment, eases its speed toward the traffic-dependent target, burns fuel, and emits periodic
telemetry.  Every event goes through a delivery heap keyed by ``emitted_at`` so that latency,
out-of-order arrival and duplicates can be injected without touching the vehicle model.
"""

from __future__ import annotations

import heapq
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import dynamics as dyn
from .config import SimulatorConfig
from .models import (EVENT_KINDS, EVENT_PERIODIC, KIND_HOME, KIND_LEISURE_OUT, KIND_LEISURE_RETURN, KIND_TO_WORK,
                     STYLE_RESPONSE_S, TELEMETRY_FIELDS, TRIP_KINDS, VEHICLE_STATES, iso_utc)
from .rng import format_event_ids, hash_keys, make_rng
from .sinks import TelemetrySink, make_sink
from .trips import choose_poi, choose_routes
from .world import World, build_world

log = logging.getLogger(__name__)

WORK_PROB = 0.65
FUEL_START_PCT_RANGE = (25.0, 95.0)
REFUEL_BELOW_PCT, REFUEL_TO_PCT = 8.0, 90.0
LATENCY_S = (0.05, 0.35)
DUP_EXTRA_DELAY_S = (0.05, 2.0)
_S_LIVE_EVENT = 21
M_PER_DEG = 111_320.0


def parse_start_time(value: str | None) -> float:
    if not value:
        return time.time()
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class LiveSimulator:
    """Vectorised state for the whole fleet; ``step`` advances one tick."""

    def __init__(self, world: World, t0: float, sink: TelemetrySink, trips_fh=None):
        self.w, self.sink, self.trips_fh = world, sink, trips_fh
        cfg = world.cfg
        self.lc, self.seed = cfg.live, cfg.seed
        self.rng = make_rng(self.seed, 99)
        n = self.n = world.fleet.n
        self.t = t0
        self.dt = self.lc.tick_seconds
        self.interval = max(self.dt, n / max(self.lc.events_per_second, 1e-9))

        f = world.fleet
        self.loc = f.home_node.copy()
        self.driving = np.zeros(n, bool)
        self.dwell_until = t0 + self.rng.uniform(0.0, 2.0 * self.interval + 30.0, n)
        self.last_emit = t0 + self.rng.uniform(0.0, self.interval, n)
        self.fuel_pct = self.rng.uniform(*FUEL_START_PCT_RANGE, n)
        self.trip_count = np.zeros(n, np.int64)
        self.event_count = np.zeros(n, np.int64)
        # per-trip state
        self.route = np.zeros(n, np.int32)
        self.seg_idx = np.zeros(n, np.int32)
        self.progress_km = np.zeros(n)
        self.speed = np.zeros(n)
        self.kind = np.zeros(n, np.int8)
        self.dest = np.zeros(n, np.int32)
        self.trip_speed_mult = np.ones(n)
        self.trip_start = np.zeros(n)
        self.trip_start_pct = np.zeros(n)
        self.trip_dist = np.zeros(n); self.trip_min = np.zeros(n); self.trip_fuel = np.zeros(n)
        # accumulators since the last emitted event
        self.acc_dist = np.zeros(n); self.acc_min = np.zeros(n); self.acc_fuel = np.zeros(n)
        self.vehicle_ids = f.ids()
        self._route_ids = world.catalog.route_ids()
        self.heap: list[tuple[float, int, dict]] = []
        self._seq = 0
        self.stats = dict(events_generated=0, events_emitted=0, duplicates=0, delayed=0, trips_completed=0)

    # ------------------------------------------------------------------ trips
    def _start_trips(self, idx: np.ndarray) -> None:
        w, f = self.w, self.w.fleet
        at_home = self.loc[idx] == f.home_node[idx]
        poi = choose_poi(f, w.catalog, idx, self.rng)
        go_work = self.rng.random(len(idx)) < WORK_PROB
        dest = np.where(at_home, np.where(go_work, f.work_node[idx], poi), f.home_node[idx])
        kind = np.where(at_home, np.where(go_work, KIND_TO_WORK, KIND_LEISURE_OUT),
                        np.where(self.loc[idx] == f.work_node[idx], KIND_HOME, KIND_LEISURE_RETURN)).astype(np.int8)
        od = w.catalog.od_lookup[self.loc[idx], dest]
        route = choose_routes(w.catalog, f, idx, od, self.seed, w.cfg.vehicles.exploration_rate, self.rng)
        if self.fuel_pct[idx].min() < REFUEL_BELOW_PCT:
            low = idx[self.fuel_pct[idx] < REFUEL_BELOW_PCT]
            self.fuel_pct[low] = REFUEL_TO_PCT
        self.route[idx], self.dest[idx], self.kind[idx] = route, dest, kind
        self.seg_idx[idx], self.progress_km[idx], self.speed[idx] = 0, 0.0, 0.0
        self.trip_speed_mult[idx] = np.exp(self.rng.normal(0.0, w.dynamics.trip_speed_sigma, len(idx)))
        self.trip_start[idx], self.trip_start_pct[idx] = self.t, self.fuel_pct[idx]
        self.trip_dist[idx] = self.trip_min[idx] = self.trip_fuel[idx] = 0.0
        self.trip_count[idx] += 1
        self.driving[idx] = True

    def _finish_trips(self, idx: np.ndarray, t_end: float) -> None:
        w = self.w
        for i in idx:
            self.stats["trips_completed"] += 1
            if self.trips_fh is not None:
                dist, mins = float(self.trip_dist[i]), float(self.trip_min[i])
                rec = {
                    "trip_id": f"LT{i + 1:06d}-{self.trip_count[i]:04d}", "vehicle_id": self.vehicle_ids[i],
                    "trip_kind": TRIP_KINDS[self.kind[i]],
                    "origin_node": w.network.node_ids[self.loc[i]], "destination_node": w.network.node_ids[self.dest[i]],
                    "route_id": self._route_ids[self.route[i]],
                    "start_time": iso_utc(self.trip_start[i]), "end_time": iso_utc(t_end),
                    "distance_km": round(dist, 4), "travel_time_min": round(mins, 4),
                    "fuel_consumed_l": round(float(self.trip_fuel[i]), 5),
                    "avg_speed_kmh": round(dist / mins * 60.0, 3) if mins > 0 else 0.0,
                    "start_fuel_pct": round(float(self.trip_start_pct[i]), 3),
                    "end_fuel_pct": round(float(self.fuel_pct[i]), 3),
                }
                self.trips_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.loc[idx] = self.dest[idx]
        self.driving[idx] = False
        self.speed[idx] = 0.0
        self.dwell_until[idx] = t_end + self.rng.uniform(*self.lc.dwell_s, len(idx))

    # ------------------------------------------------------------------- tick
    def step(self) -> list[dict]:
        """Advance one tick; return events whose delivery time has arrived."""
        w, net, cat, f, p = self.w, self.w.network, self.w.catalog, self.w.fleet, self.w.dynamics
        dt, t_now = self.dt, self.t + self.dt

        ready = np.flatnonzero(~self.driving & (self.dwell_until <= self.t))
        if len(ready):
            self._start_trips(ready)

        act = np.flatnonzero(self.driving)
        if len(act):
            seg = cat.flat_segments[cat.route_offset[self.route[act]] + self.seg_idx[act]]
            level = w.traffic.level(seg, np.full(len(act), self.t))
            noise = self.rng.standard_normal((2, len(act)))
            target = dyn.effective_speed(net.seg_baseline_speed[seg], net.seg_speed_limit[seg], level,
                                         f.speed_factor[act], f.driving_style[act], self.trip_speed_mult[act],
                                         noise[0], p)
            tau = STYLE_RESPONSE_S[f.driving_style[act]]
            spd = self.speed[act] + (target - self.speed[act]) * (1.0 - np.exp(-dt / tau))
            self.speed[act] = spd
            step_km = spd * dt / 3600.0
            fuel = dyn.fuel_litres(step_km, f.kmpl[act], np.maximum(spd, p.min_speed_kmh), level,
                                   f.driving_style[act], f.fuel_type[act], f.wear[act],
                                   noise[1], p)
            self.progress_km[act] += step_km
            for arr, val in ((self.acc_dist, step_km), (self.acc_fuel, fuel), (self.trip_dist, step_km),
                             (self.trip_fuel, fuel)):
                arr[act] += val
            self.acc_min[act] += dt / 60.0
            self.trip_min[act] += dt / 60.0
            self.fuel_pct[act] = np.maximum(self.fuel_pct[act] - fuel / f.tank_l[act] * 100.0, 0.0)

            # emit before advancing segments so the position/segment match the measurement
            due = act[(t_now - self.last_emit[act]) >= self.interval]
            if len(due):
                self._emit(due, t_now, seg[np.searchsorted(act, due)], level[np.searchsorted(act, due)])

            crossed = act[self.progress_km[act] >= net.seg_distance_km[seg]]
            if len(crossed):
                cseg = cat.flat_segments[cat.route_offset[self.route[crossed]] + self.seg_idx[crossed]]
                self.progress_km[crossed] -= net.seg_distance_km[cseg]
                self.seg_idx[crossed] += 1
                done = crossed[self.seg_idx[crossed] >= cat.route_len[self.route[crossed]]]
                if len(done):
                    self._finish_trips(done, t_now)

        self.t = t_now
        return self._release(t_now)

    def _emit(self, idx: np.ndarray, t_now: float, seg: np.ndarray, level: np.ndarray) -> None:
        net, lc = self.w.network, self.lc
        n = len(idx)
        frac = np.clip(self.progress_km[idx] / net.seg_distance_km[seg], 0.0, 1.0)
        a, b = net.seg_from[seg], net.seg_to[seg]
        lat = net.node_lat[a] + (net.node_lat[b] - net.node_lat[a]) * frac
        lon = net.node_lon[a] + (net.node_lon[b] - net.node_lon[a]) * frac
        if lc.gps_noise_m > 0:
            lat = lat + self.rng.normal(0, lc.gps_noise_m / M_PER_DEG, n)
            lon = lon + self.rng.normal(0, lc.gps_noise_m / (M_PER_DEG * np.cos(np.radians(lat))), n)
        speed = self.speed[idx]
        state = dyn.vehicle_state_code(speed, net.seg_baseline_speed[seg])
        eid = format_event_ids(hash_keys(self.seed, _S_LIVE_EVENT, idx, self.event_count[idx]))
        self.event_count[idx] += 1
        seg_ids = net.seg_ids[seg]
        latency = self.rng.uniform(*LATENCY_S, n)
        ooo = self.rng.random(n) < lc.out_of_order_rate
        ooo_delay = self.rng.uniform(*lc.out_of_order_delay_s, n)
        dup = self.rng.random(n) < lc.duplicate_rate
        dup_delay = self.rng.uniform(*DUP_EXTRA_DELAY_S, n)
        for k in range(n):
            i = idx[k]
            ev = {
                "event_id": eid[k].decode(), "vehicle_id": self.vehicle_ids[i], "timestamp": iso_utc(t_now),
                "emitted_at": None, "trip_id": f"LT{i + 1:06d}-{self.trip_count[i]:04d}",
                "segment_id": str(seg_ids[k]), "segment_seq": int(self.seg_idx[i]),
                "latitude": round(float(lat[k]), 6), "longitude": round(float(lon[k]), 6),
                "heading_deg": round(float(net.seg_heading_deg[seg[k]]), 1), "speed_kmh": round(float(speed[k]), 2),
                "traffic_level": round(float(level[k]), 3), "fuel_consumed_l": round(float(self.acc_fuel[i]), 6),
                "distance_km": round(float(self.acc_dist[i]), 5), "travel_time_min": round(float(self.acc_min[i]), 4),
                "fuel_level_pct": round(float(self.fuel_pct[i]), 3), "vehicle_state": VEHICLE_STATES[state[k]],
                "event_kind": EVENT_KINDS[EVENT_PERIODIC],
            }
            deliver = t_now + float(latency[k]) + (float(ooo_delay[k]) if ooo[k] else 0.0)
            self._push(deliver, ev)
            self.stats["events_generated"] += 1
            if ooo[k]:
                self.stats["delayed"] += 1
            if dup[k]:
                self._push(deliver + float(dup_delay[k]), dict(ev))
                self.stats["duplicates"] += 1
        self.last_emit[idx] = t_now
        self.acc_dist[idx] = self.acc_min[idx] = self.acc_fuel[idx] = 0.0

    def _push(self, deliver_at: float, ev: dict) -> None:
        ev["emitted_at"] = iso_utc(deliver_at)
        self._seq += 1
        heapq.heappush(self.heap, (deliver_at, self._seq, ev))

    def _release(self, now: float, flush: bool = False) -> list[dict]:
        out = []
        while self.heap and (flush or self.heap[0][0] <= now):
            out.append(heapq.heappop(self.heap)[2])
        return out


def run_live(cfg: SimulatorConfig, start_time: str | None = None, max_events: int | None = None,
             duration_s: float | None = None, sink: TelemetrySink | None = None) -> dict:
    """Run the live simulator until ``max_events`` / ``duration_s`` (or Ctrl-C)."""
    lc = cfg.live
    world = build_world(cfg, n_vehicles=lc.vehicles)
    t0 = parse_start_time(start_time)
    own_sink = sink is None
    sink = sink or make_sink(lc.sink, lc.output_path)
    trips_fh = None
    if lc.trips_path:
        Path(lc.trips_path).parent.mkdir(parents=True, exist_ok=True)
        trips_fh = open(lc.trips_path, "w", encoding="utf-8")
    sim = LiveSimulator(world, t0, sink, trips_fh)
    log.info("Live simulation: %d vehicles, ~%.0f events/s, start %s, speedup %s, sink %s",
             sim.n, lc.events_per_second, iso_utc(t0), lc.speedup, lc.sink)
    wall0, emitted = time.perf_counter(), 0
    try:
        while True:
            batch = sim.step()
            stop = (duration_s is not None and sim.t - t0 >= duration_s)
            if stop:
                batch += sim._release(sim.t, flush=True)
            if max_events is not None and emitted + len(batch) >= max_events:
                batch = batch[: max_events - emitted]
                stop = True
            sink.write_batch(batch)
            emitted += len(batch)
            if stop:
                break
            if lc.speedup > 0:
                lag = (sim.t - t0) / lc.speedup - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down")
    finally:
        sink.flush()
        if own_sink:
            sink.close()
        if trips_fh is not None:
            trips_fh.close()
    sim.stats["events_emitted"] = emitted
    elapsed = time.perf_counter() - wall0
    log.info("Done: %d events written (%d generated, %d duplicates, %d delayed), %d trips, %.1fs wall, %.0f ev/s",
             emitted, sim.stats["events_generated"], sim.stats["duplicates"], sim.stats["delayed"],
             sim.stats["trips_completed"], elapsed, emitted / max(elapsed, 1e-9))
    return sim.stats


if __name__ == "__main__":
    from .cli import live_main
    raise SystemExit(live_main())
