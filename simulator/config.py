"""Simulator configuration: typed dataclasses, YAML loading and CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "simulator.yaml"


def parse_hhmm(value: Any) -> float:
    """Convert ``"07:30"`` (or a number of hours) to fractional hours."""
    if isinstance(value, (int, float)):
        return float(value)
    hours, minutes = str(value).strip().split(":")[:2]
    return int(hours) + int(minutes) / 60.0


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42


@dataclass(frozen=True)
class NetworkConfig:
    rows: int = 26
    cols: int = 30
    spacing_km: float = 0.8
    jitter: float = 0.18
    diagonal_prob: float = 0.04
    edge_removal_prob: float = 0.05
    highway_every: int = 6
    arterial_every: int = 3
    express_links: int = 6
    min_segment_km: float = 0.3
    origin_lat: float = 20.0
    origin_lon: float = 78.0


@dataclass(frozen=True)
class VehicleConfig:
    count: int = 100_000
    home_zones: int = 80
    work_hubs: int = 30
    poi_count: int = 25
    min_trip_km: float = 3.0
    exploration_rate: float = 0.10


@dataclass(frozen=True)
class TrafficConfig:
    morning_start: float = 7.0
    morning_end: float = 10.0
    evening_start: float = 17.0
    evening_end: float = 20.0
    problem_segment_fraction: float = 0.02
    problem_windows: tuple[tuple[float, float], ...] = ((18.0, 20.0), (7.5, 9.5), (12.0, 14.0))
    problem_window_weights: tuple[float, ...] = (0.6, 0.25, 0.15)
    incident_rate_per_segment_day: float = 0.004
    noise_sigma: float = 0.05


@dataclass(frozen=True)
class HistoricalConfig:
    trips_per_vehicle: int = 10
    days: int = 30
    start_date: str = "2025-01-06"
    chunk_vehicles: int = 2000
    output_dir: str = "data"
    format: str = "parquet"
    compression: str = "zstd"


@dataclass(frozen=True)
class LiveConfig:
    vehicles: int = 100
    events_per_second: float = 100.0
    tick_seconds: float = 1.0
    speedup: float = 1.0
    sink: str = "stdout"
    output_path: str = "data/live_events.jsonl"
    trips_path: str = "data/live_trips.jsonl"
    duplicate_rate: float = 0.0
    out_of_order_rate: float = 0.0
    out_of_order_delay_s: tuple[float, float] = (2.0, 15.0)
    gps_noise_m: float = 3.0
    dwell_s: tuple[float, float] = (10.0, 60.0)


_TIME_FIELDS = {"morning_start", "morning_end", "evening_start", "evening_end"}


def _convert(name: str, value: Any) -> Any:
    if name in _TIME_FIELDS:
        return parse_hhmm(value)
    if name == "problem_windows":
        return tuple((parse_hhmm(a), parse_hhmm(b)) for a, b in value)
    if isinstance(value, list):
        return tuple(value)
    return value


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' in section for {cls.__name__}")
        kwargs[key] = _convert(key, value)
    return cls(**kwargs)


@dataclass(frozen=True)
class SimulatorConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vehicles: VehicleConfig = field(default_factory=VehicleConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    historical: HistoricalConfig = field(default_factory=HistoricalConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @property
    def seed(self) -> int:
        return self.simulation.seed

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SimulatorConfig":
        sections = {f.name: f.type for f in fields(cls)}
        classes = {
            "simulation": SimulationConfig, "network": NetworkConfig, "vehicles": VehicleConfig,
            "traffic": TrafficConfig, "historical": HistoricalConfig, "live": LiveConfig,
        }
        kwargs = {}
        for name, section in (data or {}).items():
            if name not in sections:
                raise ValueError(f"Unknown config section '{name}'")
            kwargs[name] = _build(classes[name], section)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SimulatorConfig":
        """Load YAML config; falls back to built-in defaults if the file is absent."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            if path == DEFAULT_CONFIG_PATH:
                return cls()
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def override(self, **dotted: Any) -> "SimulatorConfig":
        """Return a copy with ``section__field=value`` overrides (``None`` values are ignored)."""
        cfg = self
        for key, value in dotted.items():
            if value is None:
                continue
            section, name = key.split("__", 1)
            new_section = replace(getattr(cfg, section), **{name: _convert(name, value)})
            cfg = replace(cfg, **{section: new_section})
        return cfg
