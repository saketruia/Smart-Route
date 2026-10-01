import json

import pandas as pd
import pytest

from simulator.live import run_live
from simulator.models import TELEMETRY_FIELDS
from simulator.sinks import ConsoleSink, JsonlSink, ListSink, NullSink, make_sink


@pytest.fixture
def live_cfg(cfg, tmp_path):
    return cfg.override(live__vehicles=40, live__events_per_second=40, live__speedup=0,
                        live__trips_path=str(tmp_path / "trips.jsonl"))


def _run(cfg, **kw):
    sink = ListSink()
    stats = run_live(cfg, start_time="2025-01-06T17:30:00", duration_s=600, sink=sink, **kw)
    return sink.events, stats


def test_schema_and_no_disorder_by_default(live_cfg):
    ev, stats = _run(live_cfg)
    assert len(ev) > 500 and stats["duplicates"] == 0
    assert all(tuple(e) == TELEMETRY_FIELDS for e in ev[:50])
    df = pd.DataFrame(ev)
    assert df.event_id.is_unique and not df.isna().any().any()
    assert (df.speed_kmh >= 0).all() and df.traffic_level.between(0, 1).all()
    assert df.fuel_level_pct.between(0, 100).all() and (df.fuel_consumed_l >= 0).all()
    assert (pd.to_datetime(df.emitted_at) >= pd.to_datetime(df.timestamp)).all()
    assert set(df.event_kind) == {"PERIODIC"}


def test_duplicates_share_event_id(live_cfg):
    ev, stats = _run(live_cfg.override(live__duplicate_rate=0.1))
    df = pd.DataFrame(ev)
    assert stats["duplicates"] > 0 and df.event_id.duplicated().sum() == stats["duplicates"]
    dup = df[df.event_id.duplicated(keep=False)].groupby("event_id")
    assert (dup.timestamp.nunique() == 1).all()


def test_out_of_order_events(live_cfg):
    ev, stats = _run(live_cfg.override(live__out_of_order_rate=0.2))
    df = pd.DataFrame(ev)
    assert stats["delayed"] > 0
    assert pd.to_datetime(df.emitted_at).is_monotonic_increasing          # arrival order
    assert not pd.to_datetime(df.timestamp).is_monotonic_increasing       # event time disordered
    lag = (pd.to_datetime(df.emitted_at) - pd.to_datetime(df.timestamp)).dt.total_seconds()
    assert lag.max() >= live_cfg.live.out_of_order_delay_s[0]


def test_live_is_deterministic(live_cfg):
    a, _ = _run(live_cfg)
    b, _ = _run(live_cfg)
    assert a == b


def test_max_events_and_trips_file(live_cfg):
    sink = ListSink()
    run_live(live_cfg, start_time="2025-01-06T17:30:00", max_events=123, sink=sink)
    assert len(sink.events) == 123
    sink = NullSink()
    run_live(live_cfg, start_time="2025-01-06T17:30:00", duration_s=3600, sink=sink)
    lines = [json.loads(x) for x in open(live_cfg.live.trips_path)]
    assert lines and all(t["distance_km"] > 0 and t["fuel_consumed_l"] > 0 for t in lines)


def test_sinks(tmp_path, capsys):
    e = {"event_id": "EVT-1", "x": 1}
    ConsoleSink().write_batch([e])
    assert json.loads(capsys.readouterr().out) == e
    p = tmp_path / "o.jsonl"
    with JsonlSink(str(p)) as s:
        s.write_batch([e, e])
    assert len(p.read_text().splitlines()) == 2
    assert isinstance(make_sink("null"), NullSink)
    with pytest.raises(ValueError):
        make_sink("kafka")
