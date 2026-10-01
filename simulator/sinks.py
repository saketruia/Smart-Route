"""Telemetry sinks: where live events go.

A sink receives batches of event dicts (keys = ``TELEMETRY_FIELDS``).  Console and JSONL are
implemented; a Kafka or HTTP sink only has to subclass :class:`TelemetrySink` and implement
``write_batch`` (future work, deliberately not built in phase 1).
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from typing import IO, Iterable


class TelemetrySink(ABC):
    """Destination for telemetry events."""

    @abstractmethod
    def write_batch(self, events: list[dict]) -> None: ...

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "TelemetrySink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _LineSink(TelemetrySink):
    _fh: IO[str]

    def write_batch(self, events: list[dict]) -> None:
        if events:
            self._fh.write("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))

    def flush(self) -> None:
        self._fh.flush()


class ConsoleSink(_LineSink):
    """One JSON object per line on stdout."""

    def __init__(self, stream: IO[str] | None = None):
        self._fh = stream or sys.stdout


class JsonlSink(_LineSink):
    """Append events to a JSON-lines file (truncated on open)."""

    def __init__(self, path: str):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()


class NullSink(TelemetrySink):
    """Discards events (throughput tests)."""

    def __init__(self) -> None:
        self.count = 0

    def write_batch(self, events: list[dict]) -> None:
        self.count += len(events)


class ListSink(TelemetrySink):
    """Keeps events in memory (tests)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def write_batch(self, events: Iterable[dict]) -> None:
        self.events.extend(events)


def make_sink(kind: str, path: str | None = None) -> TelemetrySink:
    if kind in ("stdout", "console"):
        return ConsoleSink()
    if kind == "jsonl":
        return JsonlSink(path or "data/live_events.jsonl")
    if kind == "null":
        return NullSink()
    raise ValueError(f"Unknown sink '{kind}' (expected stdout | jsonl | null)")
