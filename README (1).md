# SmartRoute - Phase 1: Synthetic Connected-Vehicle Simulator

Generates realistic connected-vehicle data for the SmartRoute route-recommendation project:
a **historical** dataset (Parquet/CSV, up to 100K vehicles) and a **live** telemetry stream
(pluggable sinks, optional duplicates / out-of-order events). Phase 1 only: no ML, no route
optimiser, no API, no Kafka.

## 1. Setup
```bash
pip install -r requirements.txt      # numpy, pandas, pyarrow, networkx, PyYAML, pytest
```
Python 3.10+ (developed on 3.12).

## 2. Commands
| Purpose | Command |
|---|---|
| Quick dev dataset (1K vehicles x 3 trips x 14 days) | `python -m simulator.generate_history --dev` |
| Full dataset (100K vehicles x 10 trips) | `python -m simulator.generate_history` |
| Custom size | `python -m simulator.generate_history --vehicles 10000 --trips-per-vehicle 20 --output-dir data/x` |
| CSV instead of Parquet | `... --format csv` |
| Validate a dataset | `python -m simulator.validate --data-dir data` |
| Live to console | `python -m simulator.live --vehicles 50 --events-per-second 50` |
| Live to JSONL, fast, with disorder | `python -m simulator.live --sink jsonl --speedup 0 --duration 3600 --duplicate-rate 0.02 --out-of-order-rate 0.05` |
| Tests | `python -m pytest -q` |
| Single entry point | `python -m simulator {history\|live\|validate} ...` |

Every option overrides `config/simulator.yaml` (`--config` selects another file, `--seed` sets the seed).
History flags: `--vehicles --trips-per-vehicle --days --start-date --output-dir --chunk-vehicles --format --dev`.
Live flags: `--vehicles --events-per-second --sink {stdout,jsonl,null} --output --trips-output
--duplicate-rate --out-of-order-rate --speedup --tick-seconds --start-time --max-events --duration`.
`--speedup 1` is real time, `--speedup 60` is a minute per second, `--speedup 0` is as fast as possible.

## 3. Outputs (historical, in `--output-dir`, default `data/`)
| File | Content |
|---|---|
| `road_nodes` | node_id, lat, lon, zone |
| `road_segments` | segment_id, from/to node, distance, speed limit, lanes, road type, traffic profile, problem-road flag and window (ground truth) |
| `routes` | route_id, corridor (OD), kind (FASTEST/SHORTEST/HIGHWAY_PREF/...), distance, free-flow time, segment list |
| `vehicles` | vehicle_id, type, fuel type, rated km/l, driving style, speed factor, tank, home/work/favourite POI nodes |
| `trips` | trip_id, vehicle, kind, origin/destination, route_id, start/end, distance, time, fuel, avg speed |
| `telemetry` | one event per segment exit (fields below) |
| `summary.json` | the printed summary |

Telemetry fields (identical in live mode): `event_id, vehicle_id, timestamp, emitted_at, trip_id,
segment_id, segment_seq, latitude, longitude, heading_deg, speed_kmh, traffic_level, fuel_consumed_l,
distance_km, travel_time_min, fuel_level_pct, vehicle_state, event_kind`.
`timestamp` is when the measurement happened; `emitted_at` is when it was sent (they differ, and
`emitted_at` order can differ from `timestamp` order once disorder is on).
In history `event_kind = SEGMENT_COMPLETE` (distance/time/fuel cover the whole segment); in live it is
`PERIODIC` (they cover the interval since the vehicle's previous event).

## 4. How the simulation works
- **Road network** (`road_network.py`): jittered grid of ~780 nodes / ~2,960 directed segments with
  highway and arterial lines, zones (suburb/downtown/industrial), diagonals and express chords.
  Always strongly connected. Synthetic geography near lat 20, lon 78.
- **Vehicles** (`vehicles.py`): sedan/hatchback/SUV/pickup/van, petrol/diesel/hybrid, rated km/l per
  type, calm/normal/aggressive style, a latent per-vehicle wear factor, home/work/favourite POIs.
- **Traffic** (`traffic.py`): congestion in [0,1] as a pure function of (seed, segment, time): rush hours
  with direction-dependent profiles, weekend pattern, day-to-day variation, noise, incidents and
  designated "problem roads" with recurring congestion windows. Time zone is UTC.
- **Speed / time / fuel** (`dynamics.py`): speed = free-flow x congestion factor x driver factors, capped at
  130% of the limit; time = distance/speed; fuel = distance/rated km/l x penalties for low speed,
  stop-and-go, high speed and aggressive driving (hybrids get low-speed relief).
- **Routes** (`routes.py`): each home<->work / home<->POI corridor has 2-6 Dijkstra route variants.
  Drivers have habits plus 10% exploration, so about 55% of trips avoid the fastest free-flow route.
- **Trips** (`trips.py`): weekday commutes (to work morning, home evening) plus leisure trips.
- **Live** (`live.py`): stateful fleet stepping every `tick_seconds`. Vehicles cycle home -> work/leisure ->
  home with dwell time in between; speed eases toward the traffic target with a style-dependent time
  constant; fuel level persists and refuels when low; positions are interpolated along segments with GPS noise.
- **Disorder**: every event goes through a delivery heap keyed by `emitted_at`. Base latency 50-350 ms;
  `out_of_order_rate` adds a 2-15 s delay; `duplicate_rate` re-delivers the same `event_id`.
- **Sinks** (`sinks.py`): `TelemetrySink` interface with Console, JSONL, Null and in-memory List sinks.
  A Kafka/HTTP sink is one subclass with `write_batch()` (future work).
- Completed live trips are written to `live_trips.jsonl`.

Reproducibility: everything is seeded (`simulation.seed`, default 42). Same config -> identical output.

## 5. Example output
Live event:
```json
{"event_id":"EVT-a6b2793691db17ac","vehicle_id":"V000030","timestamp":"2025-01-06T17:30:02.000Z","emitted_at":"2025-01-06T17:30:02.075Z","trip_id":"LT000030-0001","segment_id":"R1918","segment_seq":0,"latitude":20.025427,"longitude":78.066522,"heading_deg":80.1,"speed_kmh":2.17,"traffic_level":0.219,"fuel_consumed_l":7.7e-05,"distance_km":0.0006,"travel_time_min":0.0167,"fuel_level_pct":61.105,"vehicle_state":"STOP_AND_GO","event_kind":"PERIODIC"}
```
Summary (100K vehicles x 10 trips): 1,000,000 trips, 19.1M events, avg trip 18.9 km / 25.8 min / 1.46 L /
44.0 km/h, peak congestion weekdays 18:00-19:00, corr(traffic, speed) = -0.62, fuel vs rated: calm 1.01,
normal 1.09, aggressive 1.31.

## 6. Performance (1 CPU, 3 GB RAM)
| Vehicles x trips | Trips | Telemetry rows | Time | Size |
|---|---|---|---|---|
| 1K x 10 | 10K | 191K | 3 s | 12 MB |
| 10K x 10 | 100K | 1.9M | 6 s | 100 MB |
| 100K x 10 | 1M | 19.1M | 39 s | 990 MB |

Live: about 27K events/s into a file sink (speedup 0, 50 vehicles).
Data is generated in chunks (`chunk_vehicles`, default 2000) and streamed to Parquet row groups, so memory stays flat.

## 7. Validation
`python -m simulator.validate` checks NaN/inf, negative values, ID uniqueness/references, timestamps and
trip-to-telemetry consistency (distance/time/fuel sums). It exits non-zero on invalid rows.
All generated datasets (100x3 through 100K x 10) pass with 0 invalid rows.

## 8. Tests
`python -m pytest -q` -> 34 tests: network connectivity and attributes, fleet ranges, traffic patterns
(rush hour, weekend, problem roads), speed/time/fuel formulas, multiple routes per corridor, telemetry
schema, seed determinism, CSV/Parquet output, validation, live sinks, duplicates and out-of-order events.

## 9. Known limitations
- Geography is synthetic; traffic is a model, not real data.
- Historical fuel level is independent per trip (trips do not chain fuel state); live mode does chain it.
- City clock is UTC; a small number of a vehicle's trips can overlap in time (the validator reports it as a warning).
- Historical output depends on `chunk_vehicles` (each chunk has its own seeded stream); fixed config -> identical output.
- Only Parquet datasets are validated. Live mode does not yet have an HTTP/Kafka sink.
- Live emission rate is a maximum: vehicles that are parked emit nothing.

## 10. Next step
Phase 2: ingestion API (FastAPI) that consumes live events, deduplicates by `event_id` and tolerates
lateness, storing to a database; then ML (travel-time / fuel prediction) on the historical dataset.
