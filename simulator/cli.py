"""Command-line entry points: history, live, validate."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import SimulatorConfig

DEV_PRESET = {"vehicles__count": 1000, "historical__trips_per_vehicle": 3, "historical__days": 14}


def _setup_logging(level: str) -> None:
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to YAML config (default: config/simulator.yaml)")
    p.add_argument("--seed", type=int, help="random seed (default from config: 42)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def history_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m simulator.generate_history",
                                description="Generate synthetic historical SmartRoute data.")
    _common(p)
    p.add_argument("--vehicles", type=int); p.add_argument("--trips-per-vehicle", type=int)
    p.add_argument("--days", type=int); p.add_argument("--start-date", help="YYYY-MM-DD (default 2025-01-06, a Monday)")
    p.add_argument("--output-dir"); p.add_argument("--chunk-vehicles", type=int)
    p.add_argument("--format", choices=["parquet", "csv"])
    p.add_argument("--dev", action="store_true", help="quick preset: 1000 vehicles x 3 trips x 14 days")
    a = p.parse_args(argv)
    _setup_logging(a.log_level)
    from .historical import format_summary, generate_history
    cfg = SimulatorConfig.load(a.config)
    if a.dev:
        cfg = cfg.override(**DEV_PRESET)
    cfg = cfg.override(
        simulation__seed=a.seed, vehicles__count=a.vehicles, historical__trips_per_vehicle=a.trips_per_vehicle,
        historical__days=a.days, historical__start_date=a.start_date, historical__output_dir=a.output_dir,
        historical__chunk_vehicles=a.chunk_vehicles, historical__format=a.format)
    print(format_summary(generate_history(cfg)))
    return 0


def live_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m simulator.live",
                                description="Run the live vehicle telemetry simulator.")
    _common(p)
    p.add_argument("--vehicles", type=int); p.add_argument("--events-per-second", type=float)
    p.add_argument("--sink", choices=["stdout", "jsonl", "null"])
    p.add_argument("--output", help="JSONL path for --sink jsonl"); p.add_argument("--trips-output")
    p.add_argument("--duplicate-rate", type=float); p.add_argument("--out-of-order-rate", type=float)
    p.add_argument("--speedup", type=float, help="simulated seconds per wall second (0 = as fast as possible)")
    p.add_argument("--tick-seconds", type=float)
    p.add_argument("--start-time", help="simulation start, ISO UTC e.g. 2025-01-06T17:30:00 (default: now)")
    p.add_argument("--max-events", type=int, help="stop after this many emitted events")
    p.add_argument("--duration", type=float, help="stop after this many simulated seconds")
    a = p.parse_args(argv)
    _setup_logging(a.log_level)
    from .live import run_live
    cfg = SimulatorConfig.load(a.config).override(
        simulation__seed=a.seed, live__vehicles=a.vehicles, live__events_per_second=a.events_per_second,
        live__sink=a.sink, live__output_path=a.output, live__trips_path=a.trips_output,
        live__duplicate_rate=a.duplicate_rate, live__out_of_order_rate=a.out_of_order_rate,
        live__speedup=a.speedup, live__tick_seconds=a.tick_seconds)
    run_live(cfg, start_time=a.start_time, max_events=a.max_events, duration_s=a.duration)
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m simulator.validate",
                                description="Validate a generated historical dataset (Parquet).")
    p.add_argument("--data-dir", default="data"); p.add_argument("--log-level", default="WARNING")
    a = p.parse_args(argv)
    _setup_logging(a.log_level)
    from .validation import format_report, validate_dataset
    report = validate_dataset(a.data_dir)
    print(format_report(report))
    return 0 if report.invalid_rows == 0 and not report.errors else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"history": history_main, "live": live_main, "validate": validate_main}
    if not argv or argv[0] not in commands:
        print("usage: python -m simulator {history|live|validate} [options]", file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])
