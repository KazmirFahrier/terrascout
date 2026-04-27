# L3 EKF-SLAM Design Note

## Purpose

L3 builds an online tree-landmark map while estimating rover pose. TerraScout uses a compact
EKF-SLAM state because the simulated orchard landmarks are point-like tree trunks and the current
mission lengths fit comfortably in memory.

Implementation: `terrascout/mapping/ekf_slam.py`

Benchmark: `benchmarks/slam_benchmark.py`

## State

The joint state is:

```text
mu = [x, y, theta, l0_x, l0_y, l1_x, l1_y, ...]^T
Sigma = joint covariance over robot pose and landmarks
```

Each landmark is a 2D tree-trunk point in the global frame.

## Prediction Model

The robot pose uses midpoint unicycle integration:

```text
theta_mid = theta + 0.5 * omega * dt
x' = x + v * cos(theta_mid) * dt
y' = y + v * sin(theta_mid) * dt
theta' = wrap(theta + omega * dt)
```

The Jacobian `G` is identity except for the pose block:

```text
d x' / d theta = -v * sin(theta_mid) * dt
d y' / d theta =  v * cos(theta_mid) * dt
```

Covariance propagation:

```text
Sigma' = G Sigma G^T + R
```

where `R` only affects the robot pose dimensions.

## Measurement Model

For landmark `l_j`:

```text
dx = l_j_x - x
dy = l_j_y - y
q = dx^2 + dy^2
range_hat = sqrt(q)
bearing_hat = wrap(atan2(dy, dx) - theta)
```

Innovation:

```text
nu = [range_measured - range_hat,
      wrap(bearing_measured - bearing_hat)]
```

The measurement Jacobian has non-zero entries only in the robot pose and selected landmark
columns:

```text
d range / d x       = -dx / sqrt(q)
d range / d y       = -dy / sqrt(q)
d range / d l_x     =  dx / sqrt(q)
d range / d l_y     =  dy / sqrt(q)
d bearing / d x     =  dy / q
d bearing / d y     = -dx / q
d bearing / d theta = -1
d bearing / d l_x   = -dy / q
d bearing / d l_y   =  dx / q
```

Kalman update:

```text
S = H Sigma H^T + Q
K = Sigma H^T S^-1
mu = mu + K nu
Sigma = (I - K H) Sigma (I - K H)^T + K Q K^T
```

The Joseph-form covariance update is used for better numerical stability.

## Landmark Association

TerraScout converts each range/bearing detection to a global point using the current pose
estimate and associates it to the nearest existing landmark if the Euclidean distance is within
`association_gate_m`. Otherwise, a new landmark is appended until `max_landmarks` is reached.

Innovations larger than the configured gate are rejected to avoid corrupting the map with row
aliases or worker detections.

## Pseudocode

```text
function predict(v, omega, dt):
    propagate robot pose with midpoint unicycle model
    build Jacobian G
    Sigma = G Sigma G^T + R

function update(detections):
    for tree detection in scan:
        landmark = nearest associated landmark
        if no landmark:
            append landmark from detection
        else:
            compute range/bearing innovation
            if innovation inside gate:
                apply EKF update
```

## Acceptance Evidence

The SLAM benchmark simulates a deterministic traversal, compares final pose and landmark map
against ground truth, and reports landmark count, mean/p95 landmark error, pose error, covariance
trace, and wall time.

## References

- Durrant-Whyte and Bailey, Simultaneous Localization and Mapping, Parts I and II.
- Thrun, Burgard, and Fox, Probabilistic Robotics, EKF-SLAM.
