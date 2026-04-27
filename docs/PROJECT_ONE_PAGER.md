# TerraScout One-Pager

## Summary

TerraScout is a simulation-first autonomy stack for a differential-drive crop-inspection rover operating in GPS-degraded orchard rows. The project demonstrates a complete autonomy slice: sensing, tracking, localization, mapping, planning, scheduling, control, visualization, and reproducible benchmarks.

## Implemented Stack

| Layer | Technique | Current implementation |
| --- | --- | --- |
| L0 Control | PID | Twin-loop waypoint tracking for differential-drive wheel commands |
| L1 Tracking | Kalman filter | Constant-velocity worker tracker from noisy lidar cluster detections |
| L2 Localization | Particle filter | Coarse-prior Monte-Carlo localization against orchard tree landmarks |
| L3 Mapping | EKF-SLAM | Compact state/covariance SLAM with range/bearing landmark updates |
| L4 Planning | Grid A* + Hybrid A* | Fast grid routing plus heading-aware Hybrid A* arc primitives |
| L5 Scheduling | Resource-aware value search | Row scheduling under priority, travel cost, battery, and daylight budgets |

## Current Metrics

Default benchmark seeds: `2, 3, 5, 7, 11`.

| Metric | Result |
| --- | ---: |
| Mission inspection success | 100% |
| Collision events | 0 |
| Mean localization error | ~0.19 m |
| EKF-SLAM landmarks in mission | 89 |
| Scheduler dropped goals | 0 |
| Mean mission wall time | ~3.0 s |

## Reproduce

```bash
python -m pip install -e ".[dev]"
python benchmarks/run_benchmark.py
python benchmarks/planner_benchmark.py
python benchmarks/slam_benchmark.py
python -m pytest
```

## Roadmap

- Use particle-filter / EKF-SLAM pose estimates directly in closed-loop control.
- Stress-test Hybrid A* across denser dynamic-obstacle scenes before making it default.
- Add battery discharge and recharge stations to the mission simulator.
- Add coverage badges, richer demo GIFs, and a short narrated demo video.

