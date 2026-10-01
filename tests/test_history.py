import pandas as pd
import pyarrow.parquet as pq

from simulator.historical import generate_history
from simulator.models import TELEMETRY_FIELDS
from simulator.validation import validate_dataset


def _gen(cfg, tmp_path, name, **kw):
    c = cfg.override(historical__output_dir=str(tmp_path / name), **kw)
    generate_history(c)
    return tmp_path / name


def test_history_schema_and_validation(cfg, tmp_path):
    d = _gen(cfg, tmp_path, "a")
    tele = pd.read_parquet(d / "telemetry.parquet")
    assert set(TELEMETRY_FIELDS) <= set(tele.columns)
    assert tele.event_id.is_unique and not tele.isna().any().any() if "emitted_at" in tele else True
    trips = pd.read_parquet(d / "trips.parquet")
    assert len(trips) == cfg.vehicles.count * cfg.historical.trips_per_vehicle
    rep = validate_dataset(d)
    assert rep.invalid_rows == 0 and not rep.errors


def test_seed_determinism(cfg, tmp_path):
    a = pd.read_parquet(_gen(cfg, tmp_path, "s1") / "telemetry.parquet")
    b = pd.read_parquet(_gen(cfg, tmp_path, "s2") / "telemetry.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs(cfg, tmp_path):
    a = pd.read_parquet(_gen(cfg, tmp_path, "s1") / "trips.parquet")
    b = pd.read_parquet(_gen(cfg, tmp_path, "s3", simulation__seed=7) / "trips.parquet")
    assert not a.equals(b)


def test_chunking_keeps_trip_count_and_validity(cfg, tmp_path):
    # Each chunk has its own seeded stream, so event details depend on chunk_vehicles,
    # but the number of trips and dataset validity must not.
    dirs = [_gen(cfg, tmp_path, "c1", historical__chunk_vehicles=100),
            _gen(cfg, tmp_path, "c2", historical__chunk_vehicles=300)]
    files = [pq.ParquetFile(d / "telemetry.parquet") for d in dirs]
    assert files[0].num_row_groups > files[1].num_row_groups
    assert len(pd.read_parquet(dirs[0] / "trips.parquet")) == len(pd.read_parquet(dirs[1] / "trips.parquet"))
    assert all(validate_dataset(d).invalid_rows == 0 for d in dirs)


def test_csv_output(cfg, tmp_path):
    d = _gen(cfg, tmp_path, "csv", historical__format="csv", vehicles__count=60)
    assert (d / "telemetry.csv").exists()
