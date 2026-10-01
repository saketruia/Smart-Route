#!/bin/sh
# Full historical dataset: 100K vehicles x 10 trips (about 19M telemetry rows, ~1 GB, ~40 s on 1 CPU).
set -e
python -m simulator.generate_history --output-dir data/full
python -m simulator.validate --data-dir data/full
