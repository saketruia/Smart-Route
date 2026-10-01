"""Data-quality validation for generated historical datasets (Parquet).

Checks every row for NaN/inf, negative or impossible values, invalid ids and impossible
timestamps, then cross-checks each trip against its telemetry (segment count, summed
distance / time / fuel, and timestamp bounds).  Trips and telemetry are read one row group
(= one generation chunk) at a time, so 100K-vehicle datasets validate in bounded memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VEHICLE_ID_RE = re.compile(r"^V\d{6,}$")
EVENT_ID_RE = r"^EVT-[0-9a-f]{16}$"
REL_TOL = 1e-3            # telemetry sums vs trip totals (telemetry is float32)
TS_TOL_MS = 2
TELEMETRY_NUMERIC = ["latitude", "longitude", "heading_deg", "speed_kmh", "traffic_level",
                     "fuel_consumed_l", "distance_km", "travel_time_min", "fuel_level_pct"]
TRIP_NUMERIC = ["total_distance_km", "total_time_min", "total_fuel_l", "average_speed_kmh"]


@dataclass
class ValidationReport:
    vehicles: int = 0
    trips: int = 0
    telemetry_rows: int = 0
    road_segments: int = 0
    invalid_vehicles: int = 0
    invalid_trips: int = 0
    invalid_telemetry: int = 0
    checks: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def invalid_rows(self) -> int:
        return self.invalid_vehicles + self.invalid_trips + self.invalid_telemetry

    def flag(self, name: str, mask: np.ndarray | pd.Series, bad: np.ndarray | None = None) -> None:
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        if n:
            self.checks[name] = self.checks.get(name, 0) + n
        if bad is not None:
            bad |= mask


def _nonfinite(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return ~np.isfinite(df[cols].to_numpy(dtype=np.float64)).all(axis=1)


def validate_dataset(data_dir: str | Path) -> ValidationReport:
    d = Path(data_dir)
    rep = ValidationReport()
    needed = ["vehicles", "road_segments", "road_nodes", "trips", "telemetry"]
    missing = [n for n in needed if not (d / f"{n}.parquet").exists()]
    if missing:
        rep.errors.append(f"Missing Parquet files in {d}: {', '.join(missing)} (only Parquet is validated)")
        return rep

    vehicles = pd.read_parquet(d / "vehicles.parquet")
    segments = pd.read_parquet(d / "road_segments.parquet")
    nodes = pd.read_parquet(d / "road_nodes.parquet")
    rep.vehicles, rep.road_segments = len(vehicles), len(segments)
    valid_veh, valid_seg, valid_node = set(vehicles.vehicle_id), set(segments.segment_id), set(nodes.node_id)

    bad = np.zeros(len(vehicles), dtype=bool)
    rep.flag("vehicle_id_format", ~vehicles.vehicle_id.map(lambda v: bool(VEHICLE_ID_RE.match(v))), bad)
    rep.flag("vehicle_id_duplicate", vehicles.vehicle_id.duplicated(), bad)
    rep.flag("vehicle_nonfinite_or_nonpositive_efficiency",
             ~np.isfinite(vehicles.fuel_efficiency_kmpl) | (vehicles.fuel_efficiency_kmpl <= 0), bad)
    rep.flag("vehicle_speed_factor_invalid", ~np.isfinite(vehicles.average_speed_factor) | (vehicles.average_speed_factor <= 0), bad)
    rep.invalid_vehicles = int(bad.sum())

    seg_bad = (segments.distance_km <= 0) | (segments.speed_limit_kmh <= 0) | ~np.isfinite(segments.baseline_speed_kmh)
    rep.flag("segment_invalid_attributes", seg_bad)
    rep.flag("segment_endpoint_unknown", ~segments.from_node.isin(valid_node) | ~segments.to_node.isin(valid_node))

    tp, ep = pq.ParquetFile(d / "trips.parquet"), pq.ParquetFile(d / "telemetry.parquet")
    if tp.num_row_groups != ep.num_row_groups:
        rep.errors.append(f"trips has {tp.num_row_groups} row groups but telemetry has {ep.num_row_groups}")
        return rep

    seen_trips: set[str] = set()
    overlaps = 0
    for g in range(tp.num_row_groups):
        t = tp.read_row_group(g).to_pandas()
        e = ep.read_row_group(g).to_pandas()
        rep.trips += len(t); rep.telemetry_rows += len(e)

        # ---- trips
        tb = np.zeros(len(t), dtype=bool)
        rep.flag("trip_nonfinite", _nonfinite(t, TRIP_NUMERIC), tb)
        rep.flag("trip_nonpositive_distance_time_or_fuel",
                 (t.total_distance_km <= 0) | (t.total_time_min <= 0) | (t.total_fuel_l <= 0), tb)
        rep.flag("trip_end_not_after_start", t.end_timestamp <= t.start_timestamp, tb)
        rep.flag("trip_vehicle_unknown", ~t.vehicle_id.isin(valid_veh), tb)
        rep.flag("trip_node_unknown", ~t.origin_node.isin(valid_node) | ~t.destination_node.isin(valid_node), tb)
        rep.flag("trip_same_origin_destination", t.origin_node.astype(str) == t.destination_node.astype(str), tb)
        rep.flag("trip_id_duplicate", t.trip_id.duplicated() | t.trip_id.isin(seen_trips), tb)
        seen_trips.update(t.trip_id)
        lens = t.route_segments.map(len)
        rep.flag("trip_segment_count_mismatch", lens != t.n_segments, tb)
        flat = pd.Series(np.concatenate(t.route_segments.to_list()))
        bad_flat = ~flat.isin(valid_seg)
        if bad_flat.any():
            per_trip = np.repeat(np.arange(len(t)), lens.to_numpy())
            rep.flag("trip_segment_unknown", np.isin(np.arange(len(t)), per_trip[bad_flat.to_numpy()]), tb)

        # ---- telemetry
        eb = np.zeros(len(e), dtype=bool)
        rep.flag("telemetry_nonfinite", _nonfinite(e, TELEMETRY_NUMERIC), eb)
        rep.flag("telemetry_negative_speed", e.speed_kmh < 0, eb)
        rep.flag("telemetry_nonpositive_distance_time_or_fuel",
                 (e.distance_km <= 0) | (e.travel_time_min <= 0) | (e.fuel_consumed_l <= 0), eb)
        rep.flag("telemetry_traffic_out_of_range", (e.traffic_level < 0) | (e.traffic_level > 1), eb)
        rep.flag("telemetry_fuel_level_out_of_range", (e.fuel_level_pct < 0) | (e.fuel_level_pct > 100), eb)
        rep.flag("telemetry_vehicle_unknown", ~e.vehicle_id.isin(valid_veh), eb)
        rep.flag("telemetry_segment_unknown", ~e.segment_id.isin(valid_seg), eb)
        rep.flag("telemetry_event_id_format", ~e.event_id.str.match(EVENT_ID_RE), eb)
        rep.flag("telemetry_event_id_duplicate", e.event_id.duplicated(), eb)
        rep.flag("telemetry_emitted_before_event", e.emitted_at < e.timestamp, eb)

        # ---- trip <-> telemetry consistency
        e["trip_id"] = e.trip_id.astype(str)
        agg = e.groupby("trip_id", observed=True).agg(
            n=("segment_seq", "size"), dist=("distance_km", "sum"), tt=("travel_time_min", "sum"),
            fuel=("fuel_consumed_l", "sum"), first=("timestamp", "min"), last=("timestamp", "max"))
        tt = t.assign(trip_id=t.trip_id.astype(str)).set_index("trip_id")
        joined = tt.join(agg, how="left")
        missing = joined["n"].isna().to_numpy()
        rep.flag("trip_without_telemetry", missing, tb)
        j = joined[~missing]
        ok_idx = np.flatnonzero(~missing)
        mism = np.zeros(len(t), dtype=bool)
        mism[ok_idx] = (
            (j.n != j.n_segments)
            | (np.abs(j.dist - j.total_distance_km) > REL_TOL * j.total_distance_km)
            | (np.abs(j.tt - j.total_time_min) > REL_TOL * j.total_time_min)
            | (np.abs(j.fuel - j.total_fuel_l) > REL_TOL * j.total_fuel_l)).to_numpy()
        rep.flag("trip_telemetry_totals_mismatch", mism, tb)
        span = np.zeros(len(t), dtype=bool)
        tol = pd.Timedelta(milliseconds=TS_TOL_MS)
        span[ok_idx] = ((j["first"] < j.start_timestamp - tol) | (j["last"] > j.end_timestamp + tol)).to_numpy()
        rep.flag("telemetry_outside_trip_window", span, tb)
        orphan = ~pd.Series(e.trip_id.unique()).isin(t.trip_id.astype(str)).to_numpy()
        if orphan.any():
            rep.flag("telemetry_orphan_trip", np.array([True] * int(orphan.sum())))

        ts = t.sort_values(["vehicle_id", "start_timestamp"])
        same_vehicle = ts.vehicle_id.astype(str).to_numpy()[1:] == ts.vehicle_id.astype(str).to_numpy()[:-1]
        overlaps += int((same_vehicle & (ts.start_timestamp.to_numpy()[1:] < ts.end_timestamp.to_numpy()[:-1])).sum())

        rep.invalid_trips += int(tb.sum()); rep.invalid_telemetry += int(eb.sum())
    if overlaps:
        rep.warnings["overlapping_trips_same_vehicle"] = overlaps
    return rep


def format_report(r: ValidationReport) -> str:
    lines = ["=" * 40, "SMARTROUTE DATA VALIDATION", "=" * 40,
             f"Vehicles:        {r.vehicles:,}", f"Road segments:   {r.road_segments:,}",
             f"Trips:           {r.trips:,}", f"Telemetry rows:  {r.telemetry_rows:,}",
             f"Invalid rows:    {r.invalid_rows:,} (vehicles {r.invalid_vehicles:,}, trips {r.invalid_trips:,}, "
             f"telemetry {r.invalid_telemetry:,})"]
    for name, n in sorted(r.checks.items()):
        lines.append(f"  FAIL {name}: {n:,}")
    for name, n in sorted(r.warnings.items()):
        lines.append(f"  WARN {name}: {n:,} (informational)")
    for err in r.errors:
        lines.append(f"  ERROR {err}")
    lines.append("RESULT: " + ("PASS" if r.invalid_rows == 0 and not r.errors else "FAIL"))
    lines.append("=" * 40)
    return "\n".join(lines)
