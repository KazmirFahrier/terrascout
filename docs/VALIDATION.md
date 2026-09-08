# Mission integrity repairs

Author: Kazmir Fahrier

## Changes in version 0.2.0

The earlier mission could report success despite crossing tree trunks. Its collision counter only covered workers, and those workers were displaced away from the rover before evaluation. An unsuccessful route search could also return a direct path to the goal. Those behaviors invalidated the previous mission safety claims.

The repair uses one shared circular footprint definition. The independent evaluator checks each executed segment against tree bodies, relative worker motion, and orchard boundaries. It reports contact entries and contact ticks separately and terminates on contact. Workers in missions receive no avoidance pose. Their motion noise has its own random stream so sensor sampling does not change the worker trajectory.

Grid search now rejects diagonal corner cutting and returns an empty path for an unreachable goal. Inflation covers the rover, obstacle, tracking margin, and grid cell uncertainty. A physically clear pose can connect to the conservative grid only through a checked segment. Hybrid search checks intermediate segments and propagates failure from its grid fallback. The follower validates lookahead and proposed motion against the known map using its navigation estimate.

Orchards now have explicit two metre headlands. The old one metre headland left insufficient room for the newly modeled rover footprint and planner margin. `headland_m` is serialized in scenario files and saved mission traces. The renderer reconstructs the saved scenario instead of silently drawing the default orchard.

Navigation defaults to the particle estimate. The same synthetic local observations feed the estimators, mapper, tracker, and worker supervisor; global detections are formed through the selected pose. Simulated wheel encoders and IMU yaw rate drive motion prediction. Truth remains an explicit diagnostic pose source and the independent evaluation reference. There is no exact worker proximity override. The mission particle likelihood uses up to ten observations with a 0.10 m residual scale, accounting for the simulator's 0.02 m range noise and 0.4 degree bearing noise. Grid paths retain their corners, while dense hybrid lookahead adds a 0.15 m clearance reserve. These changes reduce pose drift and avoid cutting into the command guard's clearance buffer.

The selected estimator's error is reported as navigation error. Particle and SLAM errors have separate fields. A truth controlled baseline therefore reports zero selected pose error instead of displaying an unrelated particle error. The raw scan generator and trunk detector remain standalone modules, which is now explicit in the architecture and portfolio documentation.

Goal completion uses all requested candidate goals as its denominator. Dropped goals remain incomplete. An empty schedule does not report success. Each visited goal requires two seconds of service; legacy row field names are documented as goal counters. This does not measure crop sensing or row coverage.

Runtime distance, time, and modeled energy are charged from actual execution, with commands limited before the remaining budgets can be exceeded. Scheduler forecasts have separate names. Merely passing a charging station no longer adds energy. There is no automatic return to charging or reserve policy.

## Reproduction corrections

The localization module benchmark now draws a fresh frame for each observation update after initial scan matching. It remains a stationary, exact map experiment over ten poses. It does not establish moving accuracy.

Tracking reports detection recall and the number of prediction samples, and missing established identities reduce continuity. Prediction error remains conditional on matched tracks in ideal constant velocity scenes. Both planners now use the same geometric heading change measurement; negative reductions are retained. Neither sparse waypoint headings nor a grid fallback establish bounded curvature execution.

The standalone SLAM benchmark runs its controller at 20 Hz and observes every 0.75 seconds. Previously it also ran control at the observation interval. It still uses ideal command odometry and true pose for the traversal controller, so its mapping accuracy is separate from the estimated pose mission result.

Package metadata is centralized in `pyproject.toml`. A constrained dependency environment, Python 3.11 and 3.12 CI jobs, an installed wheel smoke test, strict JSON output, input validation, and retained test and benchmark artifacts make failures reproducible. Performance tests run without coverage instrumentation. The mission runtime guard is now 15 seconds to include service waits and the new evaluation work; this is a simulation throughput check, not a hardware deadline.

## Regression coverage

The tests include unreachable walls, blocked corners, blocked goals, full segment clearance, a worker crossing between samples, independent worker motion, missing or stale observations, invalid inputs, empty schedules, zero simulation steps, strict JSON, actual resource accounting, a forced tree contact, and protection against global truth detections entering the mission. Existing dense covariance equivalence checks remain in place for the optimized EKF updates.

The README links the archived local reference summary and describes all benchmark populations. CI retains fresh results for each published revision. A benchmark finishing is distinct from every mission meeting its completion target; incomplete runs and contact failures remain in the CSVs.

## Remaining limitations

This is a bounded simulation improvement, not evidence of production or field readiness. It lacks raw scan worker classification, occlusion and detection failure experiments in the mission, calibrated hardware interfaces, timestamp transport, certified emergency stopping, crop inspection sensing, SLAM derived planning maps, and guarantees of recovery from worker encounters or estimator drift. The exact scheduler's straight line resource model can be optimistic even though runtime limits are enforced.
