# TerraScout One-Pager

## Summary

TerraScout is a simulation-first autonomy stack for a differential-drive crop-inspection rover operating in GPS-degraded orchard rows. The project demonstrates a complete autonomy slice: sensing, tracking, localization, mapping, planning, scheduling, control, visualization, and reproducible benchmarks.

## Implemented Stack

| Layer | Technique | Current implementation |
| --- | --- | --- |
| L0 Control | PID | Twin-loop waypoint tracking for differential-drive wheel commands |
| L1 Tracking | Kalman filter | Constant-velocity worker tracker from noisy lidar cluster detections |
| L2 Localization | KLD-adaptive particle filter | Coarse-to-fine MCL relocalization against orchard tree landmarks |
| L3 Mapping | EKF-SLAM | Compact state/covariance SLAM with range/bearing landmark updates |
| L4 Planning | Grid A* + Hybrid A* | Fast grid routing plus heading-aware arc primitives and analytic connector |
| L5 Scheduling | Resource-aware value search | Row scheduling under priority, travel cost, battery, and daylight budgets |
| L6 Battery | Energy accounting | State-of-charge drain and recharge-station contact metrics |
| L7 Safety | Command supervision | Wheel-command scaling near perceived or predicted workers |
| Sensors | Lidar + IMU + encoders | 270-degree / 0.5-degree lidar scans, yaw-rate samples, and wheel-encoder ticks |

## Current Metrics

Default benchmark seeds: `2, 3, 5, 7, 11`.

| Metric | Result |
| --- | ---: |
| Mission inspection success | 100% |
| Collision events | 0 |
| PID straight-line cross-track | <0.03 m max over 10 slip runs |
| PID 90-degree heading step | <1.5 s settle, <8% overshoot |
| 10-worker tracking prediction | <0.05 m mean 1-second error |
| 10-worker association accuracy | 100% on deterministic benchmark |
| Particle-filter relocalization | <0.03 m p95 from +/-5 m, +/-30 degree prior |
| Scheduler oracle gap | 0% on deterministic 7-goal benchmark |
| Resource scheduler oracle gap | 0% across 50 constrained randomized layouts |
| Hybrid A* steering effort | >80% lower than grid A* baseline |
| EKF-SLAM map accuracy | <0.10 m mean landmark error on deterministic traversal |
| Mean localization error | ~0.19 m |
| EKF-SLAM landmarks in mission | 89 |
| Scheduler dropped goals | 0 |
| 30-row acceptance pass | 10/10 priority goals, 0 collisions, 0.199 m mean pose error |
| 30-row wall time | 12.43 s max over 20 seeded missions |
| Final battery SOC | ~90% |
| Mean mission wall time | ~3.0 s |
| Runtime safety layer | Reports interventions, stops, and minimum perceived worker clearance |

The mission runner can use ground-truth pose, particle-filter pose, or EKF-SLAM pose for planning and waypoint control via `--pose-source truth|particle|slam`.

The stress benchmark currently runs worker-present grid/truth and grid/particle modes plus clear-lane grid/SLAM and Hybrid A*/SLAM modes across seeds `2, 7, 11`; all four modes complete with 100% success and zero collisions.

Per-layer design notes live in [docs/design](design/README.md), with reproducible PDF rendering
via `python docs/design/render_design_pdfs.py`. This one-pager renders to
`docs/PROJECT_ONE_PAGER.pdf` via `python docs/render_project_one_pager_pdf.py`.

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m terrascout.runner.reproduce --skip-gif
```

The reproduce command writes the mission trace, PNG, metrics CSVs, benchmark CSVs, stress-test outputs, and `artifacts/reproduce_summary.json`. Drop `--skip-gif` to regenerate the animated mission GIF too.

Individual commands:

```bash
python -m terrascout.runner.mission --scenario scenarios/default_orchard.json
python benchmarks/run_benchmark.py
python benchmarks/control_benchmark.py
python benchmarks/tracking_benchmark.py
python benchmarks/localization_benchmark.py
python benchmarks/scheduler_benchmark.py
python benchmarks/resource_scheduler_benchmark.py
python benchmarks/planner_benchmark.py
python benchmarks/slam_benchmark.py
python benchmarks/end_to_end_benchmark.py
python benchmarks/stress_benchmark.py
python docs/design/render_design_pdfs.py
python docs/render_project_one_pager_pdf.py
python -m pytest
```

## Roadmap

- Stress-test KLD-adaptive particle-filter and EKF-SLAM closed-loop control across larger randomized scenario suites.
- Stress-test Hybrid A* across denser dynamic-obstacle scenes before making it default.
- Add coverage badges, richer demo GIFs, and a short narrated demo video.
