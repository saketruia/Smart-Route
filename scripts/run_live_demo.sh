#!/bin/sh
# Real-time demo: 100 vehicles, ~100 events/s, JSONL sink, with duplicates + out-of-order arrival.
python -m simulator.live --vehicles 100 --events-per-second 100 --sink jsonl \
  --output data/live_events.jsonl --duplicate-rate 0.02 --out-of-order-rate 0.05 "$@"
