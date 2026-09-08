# TerraScout

## Summary

An orchard robotics simulation by Kazmir Fahrier. Estimated pose drives goal service missions with independent swept collision checks. The project demonstrates software integration; hardware, field accuracy, and production safety remain unvalidated.

## Implemented Stack

| Layer | Technique | Current implementation |
| --- | --- | --- |
| Control | PID feedback | Measured speed feedback and checked waypoint lookahead |
| Tracking | Kalman filter | Synthetic worker observations transformed by selected pose |
| Localization | Particle filter | Default navigation against a known tree map |
| Mapping | EKF SLAM | Alternate pose source and independent error reporting |
| Planning | Grid and Hybrid A* | Inflated footprints, checked segments, explicit failures |
| Scheduling | Exact resource search | Goal priorities and separately enforced execution budgets |
| Evaluation | Swept contact checks | Trees, independent workers, and orchard boundaries |
| Sensors | Synthetic observations | Local labeled centroids, encoders, and IMU yaw rate |

## Current Evidence

| Metric | Result |
| --- | --- |
| Default navigation | Particle filter |
| Goal completion | Includes all requested goals in denominator |
| Collision evaluation | Full body and swept motion |
| Resource accounting | Actual distance, elapsed time, modeled energy |
| Benchmark evidence | README and archived reference summary |
| Raw scan perception | Separate modules; not integrated in mission |
| Field validation | Not performed |
| Authorship | Kazmir Fahrier |

## Reproduce

```bash
python -m pip install -e ".[dev]" -c constraints-ci.txt
terrascout-reproduce --skip-gif
python -m pytest
```

## Roadmap

* Integrate raw scan perception and explicit detection failures.
* Validate calibrated hardware sensors and actuators.
* Evaluate independent worker encounters and recovery policies.
