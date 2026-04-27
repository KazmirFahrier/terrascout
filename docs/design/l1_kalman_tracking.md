# L1 Kalman Tracking Design Note

## Purpose

L1 tracks moving workers detected as lidar cluster centroids and predicts their positions one
second ahead for safety supervision and obstacle inflation.

Implementation: `terrascout/tracking/kalman.py`

Benchmark: `benchmarks/tracking_benchmark.py`

## State And Measurements

Each worker track uses a constant-velocity state:

```text
x = [p_x, p_y, v_x, v_y]^T
```

Each lidar cluster measurement observes position only:

```text
z = [p_x, p_y]^T
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

## Prediction Model

For timestep `dt`:

```text
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]

x' = F x
P' = F P F^T + Q
```

The process covariance is diagonal and scaled by the tracker process-noise setting:

```text
Q = q * diag(0.25 * dt^4, 0.25 * dt^4, dt^2, dt^2)
```

This is a compact approximation of white acceleration noise.

## Update Model

The linear Kalman update is:

```text
y = z - H x'
S = H P' H^T + R
K = P' H^T S^-1
x = x' + K y
P = (I - K H) P'
```

where `R = sigma_z^2 I`.

## Data Association

The multi-object tracker predicts all active tracks, builds all measurement-track pairs within
`gate_m`, sorts by Euclidean distance, then greedily accepts non-conflicting nearest pairs.
Unmatched measurements create new tracks. Unmatched tracks accumulate `missed` counts and are
removed after `max_missed` updates.

## Prediction Output

For a safety horizon `tau`, the tracker emits:

```text
p_x(t + tau) = p_x + v_x * tau
p_y(t + tau) = p_y + v_y * tau
```

These predicted points are used by both the planner obstacle map and the command-level safety
supervisor.

## Pseudocode

```text
function update(detections, dt):
    measurements = worker detections as xy vectors
    for track in tracks:
        track.predict(dt)
    candidate_pairs = all gated track-measurement distances
    for pair in candidate_pairs sorted by distance:
        if track and measurement are still unmatched:
            track.update(measurement)
    mark unmatched tracks missed
    create tracks for unmatched measurements
    prune tracks whose missed count is too high
```

## Acceptance Evidence

The L1 benchmark runs 10 moving workers across deterministic scenes and reports mean 1-second
prediction error plus association accuracy. Current tests require prediction error below
0.20 m and stable ID association on seeded scenes.

## References

- Thrun, Burgard, and Fox, Probabilistic Robotics, Kalman filtering.
- Bar-Shalom, Li, and Kirubarajan, Estimation with Applications to Tracking and Navigation.
