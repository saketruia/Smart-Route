# SmartRoute — Quick Demo MVP

This is the fast submission-ready prototype built on top of the existing Phase 1 simulator.

## Implemented

- Synthetic connected-vehicle historical dataset
- Historical traffic/time/fuel aggregation by road segment
- Directed road graph
- Explainable multi-objective route scoring
- Dijkstra route selection
- Alternative route comparison: shortest, fastest and lowest-fuel
- FastAPI service
- Browser dashboard with an interactive SVG road map
- Configurable priority: balanced, time, fuel or distance
- Historical hour selection to demonstrate rush-hour effects

## Not implemented in this MVP

ML prediction, Kafka, PostgreSQL, Redis, Kubernetes, vector search and cloud deployment are intentionally deferred. The architecture can be extended with them later.

## Run on Windows PowerShell

From the repository root:

```powershell
pip install -r requirements.txt
pip install -r requirements-demo.txt
python scripts/run_app.py
```

Open:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health

API docs:

http://127.0.0.1:8000/docs

The API reads the existing files in `data/`. Generate the dev dataset first if those files are missing:

```powershell
python -m simulator.generate_history --dev
python -m simulator.validate --data-dir data
```

## Demo story

1. Open the dashboard.
2. Select a source and destination.
3. Start with `Balanced`.
4. Use hour `18` to demonstrate the simulated evening traffic pattern.
5. Click **Find SmartRoute**.
6. Point out that the recommendation is based on a weighted combination of historical time, fuel, congestion and distance.
7. Switch between `Time`, `Fuel` and `Shortest Distance` to show that the selected path changes with the user's objective.
8. Compare the recommendation with the alternative routes shown below the map.

## Architecture

Phase 1 simulator -> Parquet telemetry -> historical segment analytics -> weighted graph -> Dijkstra -> FastAPI -> dashboard.

The routing layer is deliberately deterministic and explainable, which makes the demo easy to validate and present.
