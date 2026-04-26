# TerraScout

TerraScout is a compact autonomy demo for a simulated crop-inspection rover in a GPS-degraded orchard. It was scoped for a short creator-challenge build: make the rover actually move, inspect rows, avoid obvious hazards, emit metrics, and leave a clean path for deeper robotics modules later.

![TerraScout orchard inspection animation](docs/mission_trace.gif)

## What Works Today

- Procedural orchard generation with tree landmarks and moving field workers.
- Differential-drive rover dynamics with wheel-speed saturation and slip.
- Twin-loop PID waypoint tracking.
- Lidar-style noisy cluster detections for trees and workers.
- Constant-velocity Kalman tracking for moving worker detections.
- Particle-filter localization against orchard tree landmarks from a coarse pose prior.
- Online Gaussian tree-landmark mapping from local range/bearing detections.
- Grid A* path planning over inflated tree and worker obstacles.
- Value-iteration inspection scheduler over row-goal priority and travel cost.
- End-to-end row-inspection mission runner with deterministic metrics.
- Static PNG and animated GIF rendering for mission traces.
- Benchmark CSV generation, unit tests, and GitHub Actions CI.

This is intentionally **TerraScout MVP**, not a finished research-grade autonomy stack. The current mission runner still uses ground-truth pose for closed-loop control and a grid planner for routing; full EKF-SLAM and Hybrid A* are on the roadmap.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m terrascout.runner.mission --seed 7 --trace artifacts/mission_trace.json
python -m terrascout.viz.render --trace artifacts/mission_trace.json --out artifacts/mission_trace.png --gif artifacts/mission_trace.gif
python benchmarks/run_benchmark.py
python -m pytest
```

If `pytest` is not installed, the tests also run with the standard library:

```bash
python -m unittest discover -s tests
```

## Current Benchmark

Run on a local laptop with the default MVP configuration: 8 tree rows, 7 inspection lanes, 14 trees per row, and one moving worker.

| Seeds | Mean inspection success | Collision events | Mean localization error | Mean wall time |
| --- | ---: | ---: | ---: | ---: |
| 2, 3, 5, 7, 11 | 100% | 0 | ~0.19 m | ~2.0 s |

Benchmark output is written to `artifacts/benchmark.csv`.

## Architecture

```text
terrascout/
  sim/        orchard world, rover kinematics, sensor detections
  control/    PID drive controller
  tracking/   Kalman worker tracker
  localize/   particle-filter localization
  mapping/    online tree-landmark mapper
  plan/       grid A* planner
  scheduler/  value-iteration inspection scheduler
  runner/     end-to-end mission loop
  viz/        mission trace renderer
```

Runtime flow:

1. The world emits noisy lidar-style detections.
2. The Kalman tracker updates worker tracks and predicts near-future positions.
3. The particle filter estimates rover pose from local tree observations.
4. The landmark mapper accumulates a tree map from range/bearing detections.
5. The scheduler chooses the next inspection goal from travel cost and row priority.
6. The planner builds an inflated occupancy grid from trees and predicted workers.
7. The PID controller tracks the next waypoint.
8. The mission runner records inspection, collision, mapping, localization, path-length, and timing metrics.

## Roadmap

- Use the particle-filter pose estimate directly in closed-loop control.
- Replace the lightweight landmark mapper with EKF-SLAM.
- Replace grid A* with Hybrid A* over `(x, y, theta)`.
- Extend the scheduler with battery/time/priority state.
- Generate animated GIFs for challenge/demo submissions.
- Expand tests into coverage-gated CI.

## Why This Exists

The goal is to show an end-to-end autonomy slice that is small enough to understand but complete enough to run: a simulated rover, sensors, tracking, planning, control, evaluation, and a reproducible public repo.
