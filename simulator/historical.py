"""Historical dataset generation (chunked, constant memory).

Vehicles are processed in blocks of ``chunk_vehicles``.  For each block we plan trips,
simulate every segment traversal, and append the resulting Arrow tables to Parquet (or CSV)
files.  Trips and telemetry share chunk boundaries (one Parquet row group per chunk), which
lets the validator check per-trip consistency without loading whole files.

Output (``output_dir``):
    road_nodes, road_segments, routes, vehicles   (static)
    trips, telemetry                              (chunked)
    summary.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SimulatorConfig
from .models import DRIVING_STYLES, PROFILE_NAMES
from .rng import make_rng
from .telemetry import TELEMETRY_SCHEMA, TRIPS_SCHEMA, TableBuilder, simulate_traversals
from .traffic import TrafficEngine
from .trips import TripPlan, plan_trips
from .world import World, build_world

log = logging.getLogger(__name__)
MIN_SEGMENT_SAMPLES = 20


class TableWriter:
    """Incremental writer for one logical table (Parquet row-group per chunk, or CSV)."""

    def __init__(self, path: Path, schema: pa.Schema, fmt: str, compression: str):
        self.fmt, self.rows = fmt, 0
        self.path = path.with_suffix(".parquet" if fmt == "parquet" else ".csv")
        if fmt == "parquet":
            self._pq = pq.ParquetWriter(self.path, schema, compression=compression)
        elif fmt == "csv":
            self._fh, self._header = self.path.open("w", newline=""), True
        else:
            raise ValueError(f"Unsupported format '{fmt}' (use parquet or csv)")

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self.fmt == "parquet":
            self._pq.write_table(table, row_group_size=table.num_rows)
        else:
            df = _csv_frame(table)
            df.to_csv(self._fh, header=self._header, index=False)
            self._header = False
        self.rows += table.num_rows

    def close(self) -> None:
        (self._pq if self.fmt == "parquet" else self._fh).close()


def _csv_frame(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
        elif df[col].dtype == object and len(df) and isinstance(df[col].iloc[0], (list, np.ndarray)):
            df[col] = df[col].map(lambda v: "|".join(v))
    return df


def write_static(name: str, data: pd.DataFrame | pa.Table, out: Path, fmt: str) -> Path:
    """Write a small static table."""
    path = out / f"{name}.{'parquet' if fmt == 'parquet' else 'csv'}"
    if fmt == "parquet":
        pq.write_table(data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False), path)
    else:
        _csv_frame(data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False)).to_csv(path, index=False)
    return path


class _Corr:
    """Running Pearson correlation."""

    def __init__(self) -> None:
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n += len(x); self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    def value(self) -> float:
        cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n)
        vx = self.sxx / self.n - (self.sx / self.n) ** 2
        vy = self.syy / self.n - (self.sy / self.n) ** 2
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


class SummaryAccumulator:
    """Accumulates dataset statistics chunk by chunk."""

    def __init__(self, world: World):
        self.w = world
        n_seg = world.network.n_segments
        self.trips = self.events = 0
        self.dist = self.time = self.fuel = 0.0
        self.seg_sum, self.seg_cnt = np.zeros(n_seg), np.zeros(n_seg)
        self.hour_sum, self.hour_cnt = np.zeros(24), np.zeros(24)
        self.wk_hour_sum, self.wk_hour_cnt = np.zeros(24), np.zeros(24)
        self.corr = {k: _Corr() for k in ("traffic~speed", "traffic~min_per_km", "traffic~l_per_km")}
        self.style_ratio, self.style_cnt = np.zeros(3), np.zeros(3)
        self.od_routes: set[tuple[int, int]] = set()
        self.off_best = 0

    def update(self, plan: TripPlan, tb) -> None:
        w = self.w
        self.trips += len(plan); self.events += len(tb.seg)
        self.dist += tb.trip_dist.sum(); self.time += tb.trip_time_min.sum(); self.fuel += tb.trip_fuel.sum()
        np.add.at(self.seg_sum, tb.seg, tb.level); np.add.at(self.seg_cnt, tb.seg, 1)
        _, hour, weekend = TrafficEngine.time_parts(tb.t_exit)
        h = hour.astype(int)
        self.hour_sum += np.bincount(h, tb.level, 24); self.hour_cnt += np.bincount(h, minlength=24)
        wk = ~weekend
        self.wk_hour_sum += np.bincount(h[wk], tb.level[wk], 24); self.wk_hour_cnt += np.bincount(h[wk], minlength=24)
        self.corr["traffic~speed"].update(tb.level, tb.speed)
        self.corr["traffic~min_per_km"].update(tb.level, tb.tt_min / tb.dist)
        self.corr["traffic~l_per_km"].update(tb.level, tb.fuel / tb.dist)
        v_of = plan.vehicle_idx[tb.trip_of]
        ratio = tb.fuel * w.fleet.kmpl[v_of] / tb.dist          # fuel relative to rated efficiency
        style = w.fleet.driving_style[v_of]
        self.style_ratio += np.bincount(style, ratio, 3); self.style_cnt += np.bincount(style, minlength=3)
        self.od_routes.update(zip(plan.od.tolist(), plan.route.tolist()))
        self.off_best += int((plan.route != w.catalog.od_best_ff_route[plan.od]).sum())

    def result(self) -> dict:
        net = self.w.network
        seg_mean = np.divide(self.seg_sum, self.seg_cnt, out=np.zeros_like(self.seg_sum), where=self.seg_cnt > 0)
        eligible = np.where(self.seg_cnt >= MIN_SEGMENT_SAMPLES, seg_mean, -1.0)
        peak_seg = int(np.argmax(eligible if eligible.max() >= 0 else seg_mean))
        wk_mean = np.divide(self.wk_hour_sum, self.wk_hour_cnt, out=np.zeros(24), where=self.wk_hour_cnt > 0)
        peak_hour = int(np.argmax(wk_mean))
        od_used = {}
        for od, route in self.od_routes:
            od_used.setdefault(od, set()).add(route)
        multi = sum(1 for r in od_used.values() if len(r) > 1)
        style_mean = np.divide(self.style_ratio, self.style_cnt, out=np.zeros(3), where=self.style_cnt > 0)
        return {
            "vehicles": self.w.fleet.n, "trips": self.trips, "telemetry_events": self.events,
            "road_nodes": net.n_nodes, "road_segments": net.n_segments,
            "avg_trip_distance_km": self.dist / self.trips, "avg_trip_time_min": self.time / self.trips,
            "avg_trip_fuel_l": self.fuel / self.trips, "avg_trip_speed_kmh": self.dist / (self.time / 60.0),
            "mean_traffic_level": float(self.seg_sum.sum() / self.seg_cnt.sum()),
            "peak_traffic_segment": str(net.seg_ids[peak_seg]), "peak_traffic_segment_mean": float(seg_mean[peak_seg]),
            "peak_traffic_segment_profile": PROFILE_NAMES[int(net.seg_profile[peak_seg])],
            "peak_traffic_segment_is_problem_road": bool(net.problem_mask[peak_seg]),
            "peak_congestion_hour_weekday": f"{peak_hour:02d}:00-{(peak_hour + 1) % 24:02d}:00",
            "mean_traffic_by_weekday_hour": [round(float(x), 3) for x in wk_mean],
            "correlations": {k: round(c.value(), 3) for k, c in self.corr.items()},
            "fuel_vs_rated_by_style": {DRIVING_STYLES[i]: round(float(style_mean[i]), 3) for i in range(3)},
            "corridors_in_catalog": self.w.catalog.n_od,
            "corridors_used": len(od_used), "corridors_with_multiple_routes_used": multi,
            "share_trips_not_on_fastest_free_flow_route": self.off_best / self.trips,
            "problem_road_count": int(net.problem_mask.sum()),
        }


def format_summary(s: dict) -> str:
    bar = "=" * 40
    c, st = s["correlations"], s["fuel_vs_rated_by_style"]
    lines = [
        bar, "SMARTROUTE SIMULATION SUMMARY", bar, "",
        f"Vehicles:                 {s['vehicles']:,}", f"Trips:                    {s['trips']:,}",
        f"Telemetry events:         {s['telemetry_events']:,}", f"Road segments:            {s['road_segments']:,}",
        f"Road nodes:               {s['road_nodes']:,}", "",
        f"Average trip distance:    {s['avg_trip_distance_km']:.1f} km",
        f"Average trip time:        {s['avg_trip_time_min']:.1f} min",
        f"Average trip fuel:        {s['avg_trip_fuel_l']:.2f} L",
        f"Average trip speed:       {s['avg_trip_speed_kmh']:.1f} km/h",
        f"Mean traffic level:       {s['mean_traffic_level']:.2f}", "",
        f"Peak traffic segment:     {s['peak_traffic_segment']} (mean {s['peak_traffic_segment_mean']:.2f}, "
        f"{s['peak_traffic_segment_profile']}, problem road: {s['peak_traffic_segment_is_problem_road']})",
        f"Peak congestion hour:     {s['peak_congestion_hour_weekday']} (weekdays)", "",
        "Correlations (per segment traversal):",
        f"  traffic ~ speed         {c['traffic~speed']:+.2f}",
        f"  traffic ~ min per km    {c['traffic~min_per_km']:+.2f}",
        f"  traffic ~ litres per km {c['traffic~l_per_km']:+.2f}",
        "Fuel used vs rated efficiency (1.00 = rated):",
        f"  calm {st['calm']:.2f} | normal {st['normal']:.2f} | aggressive {st['aggressive']:.2f}", "",
        f"Route diversity: {s['corridors_with_multiple_routes_used']:,} of {s['corridors_used']:,} used corridors "
        f"saw more than one route; {100 * s['share_trips_not_on_fastest_free_flow_route']:.0f}% of trips "
        "were not on the fastest free-flow route.", bar,
    ]
    return "\n".join(lines)


def generate_history(cfg: SimulatorConfig) -> dict:
    """Generate the full historical dataset. Returns the summary dict."""
    h = cfg.historical
    out = Path(h.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    world = build_world(cfg)
    builder = TableBuilder(world)
    write_static("road_nodes", world.network.nodes_frame(), out, h.format)
    write_static("road_segments", world.network.segments_frame(), out, h.format)
    write_static("routes", world.catalog.routes_frame(world.network), out, h.format)
    write_static("vehicles", world.fleet.to_frame(world.network.node_ids), out, h.format)

    trips_w = TableWriter(out / "trips", TRIPS_SCHEMA, h.format, h.compression)
    tel_w = TableWriter(out / "telemetry", TELEMETRY_SCHEMA, h.format, h.compression)
    acc = SummaryAccumulator(world)
    n, chunk = cfg.vehicles.count, h.chunk_vehicles
    n_chunks = -(-n // chunk)
    try:
        for chunk_id, a in enumerate(range(0, n, chunk)):
            b = min(a + chunk, n)
            plan = plan_trips(world.fleet, a, b, world.catalog, h, cfg.vehicles, cfg.seed, chunk_id)
            tb = simulate_traversals(world, plan, make_rng(cfg.seed, 5, chunk_id))
            trips_w.write(builder.trips(plan, tb, a, b))
            tel_w.write(builder.telemetry(plan, tb, a, b, make_rng(cfg.seed, 6, chunk_id)))
            acc.update(plan, tb)
            if chunk_id % max(1, n_chunks // 20) == 0 or chunk_id == n_chunks - 1:
                el = time.perf_counter() - t0
                log.info("chunk %d/%d | %s trips | %s events | %.0fs elapsed", chunk_id + 1, n_chunks,
                         f"{acc.trips:,}", f"{acc.events:,}", el)
    finally:
        trips_w.close(); tel_w.close()

    elapsed = time.perf_counter() - t0
    summary = acc.result()
    summary["generation_seconds"] = round(elapsed, 1)
    summary["events_per_second_generated"] = round(acc.events / elapsed)
    summary["output_bytes"] = {p.name: p.stat().st_size for p in sorted(out.glob("*.*")) if p.suffix in (".parquet", ".csv")}
    summary["config"] = {k: asdict(v) for k, v in vars(cfg).items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Done in %.1fs -> %s", elapsed, out.resolve())
    return summary
