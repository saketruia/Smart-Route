#!/bin/sh
# Quick end-to-end check: small dataset, validate, short live demo.
set -e
python -m simulator.generate_history --dev --output-dir data/dev
python -m simulator.validate --data-dir data/dev
python -m simulator.live --vehicles 20 --events-per-second 20 --speedup 0 --max-events 20 \
  --duplicate-rate 0.1 --out-of-order-rate 0.2 --start-time 2025-01-06T17:30:00
