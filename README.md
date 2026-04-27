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
- Compact EKF-SLAM back end with state expansion, covariance propagation, and range/bearing updates.
- Grid A* path planning over inflated tree and worker obstacles.
- Hybrid A* planning over a coarse `(x, y, theta)` lattice with forward/reverse arc primitives.
- Resource-aware inspection scheduler over row priority, travel cost, battery, and daylight budgets.
- End-to-end row-inspection mission runner with deterministic metrics.
- Static PNG and animated GIF rendering for mission traces.
- Benchmark CSV generation, unit tests, and GitHub Actions CI.

This is intentionally **TerraScout MVP**, not a finished research-grade autonomy stack. The default mission uses ground-truth pose for the most stable demo path, but estimated-pose control is available with `--pose-source particle` or `--pose-source slam`. Hybrid A* is available with `--planner hybrid`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m terrascout.runner.mission --seed 7 --trace artifacts/mission_trace.json
python -m terrascout.runner.mission --seed 7 --planner hybrid --trace artifacts/hybrid_trace.json
python -m terrascout.runner.mission --seed 7 --pose-source particle --trace artifacts/particle_trace.json
python -m terrascout.viz.render --trace artifacts/mission_trace.json --out artifacts/mission_trace.png --gif artifacts/mission_trace.gif
python benchmarks/run_benchmark.py
python benchmarks/planner_benchmark.py
python benchmarks/slam_benchmark.py
python benchmarks/stress_benchmark.py
python -m pytest
```

If `pytest` is not installed, the tests also run with the standard library:

```bash
python -m unittest discover -s tests
```

## Current Benchmark

Run on a local laptop with the default MVP configuration: 8 tree rows, 7 inspection lanes, 14 trees per row, and one moving worker.

| Seeds | Pose source | Mean inspection success | Collision events | Mean localization error | Scheduler drops | Mean wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2, 3, 5, 7, 11 | truth | 100% | 0 | ~0.19 m | 0 | ~3.0 s |

Benchmark output is written to `artifacts/benchmark.csv`.

Planner benchmark output is written to `artifacts/planner_benchmark.csv`. On the same local run, grid A* averaged ~9 ms per plan and Hybrid A* averaged ~55 ms per plan while returning sparse heading-aware pose paths.

SLAM benchmark output is written to `artifacts/slam_benchmark.csv`. The compact EKF-SLAM benchmark observes about 49 tree landmarks in ~2.5 ms per seeded run.

Stress benchmark output is written to `artifacts/stress_benchmark_summary.csv`. The current stress suite covers grid/truth, grid/particle, grid/SLAM, and Hybrid A*/SLAM across seeds `2, 7, 11`; all four modes currently complete with 100% success and zero collisions.

## Architecture

```text
terrascout/
  sim/        orchard world, rover kinematics, sensor detections
  control/    PID drive controller
  tracking/   Kalman worker tracker
  localize/   particle-filter localization
  mapping/    online mapper and compact EKF-SLAM
  plan/       grid A* and Hybrid A* planners
  scheduler/  value-iteration inspection scheduler
  runner/     end-to-end mission loop
  viz/        mission trace renderer
```

Runtime flow:

1. The world emits noisy lidar-style detections.
2. The Kalman tracker updates worker tracks and predicts near-future positions.
3. The particle filter estimates rover pose from local tree observations.
4. The landmark mapper accumulates a tree map from range/bearing detections.
5. The scheduler chooses the next inspection goal from travel cost, row priority, battery, and daylight budgets.
6. The planner builds an inflated occupancy grid from trees and predicted workers.
7. The PID controller tracks the next waypoint using truth, particle-filter, or EKF-SLAM pose.
8. The mission runner records inspection, collision, mapping, EKF-SLAM, localization, path-length, and timing metrics.

## Roadmap

- Stress-test estimated-pose control across larger randomized scenario suites.
- Use Hybrid A* as the default mission planner after more stress testing.
- Extend the scheduler with battery/time/priority state.
- Expand tests into coverage-gated CI.

## Why This Exists

The goal is to show an end-to-end autonomy slice that is small enough to understand but complete enough to run: a simulated rover, sensors, tracking, planning, control, evaluation, and a reproducible public repo.

For a reviewer-friendly summary, see [docs/PROJECT_ONE_PAGER.md](docs/PROJECT_ONE_PAGER.md).
