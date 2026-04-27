# L2 MCL Localization Design Note

## Purpose

L2 estimates rover pose in a GPS-degraded orchard using local lidar tree detections and a prior
tree map. It combines a coarse-to-fine scan-match reset with KLD-adaptive particle-filter updates.

Implementation: `terrascout/localize/particle.py`

Benchmark: `benchmarks/localization_benchmark.py`

## State And Inputs

Each particle stores:

```text
x_i = [p_x, p_y, theta]
w_i = particle weight
```

Measurements are local range/bearing detections of tree trunks:

```text
z_j = [r_j, beta_j]
```

The map is a set of global tree landmarks:

```text
M = {m_k = [x_k, y_k]}
```

## Motion Prediction

Given commanded linear and angular velocity, each particle is propagated with noisy unicycle
motion:

```text
v_i = v + eps_v
omega_i = omega + eps_omega
theta_mid = theta_i + 0.5 * omega_i * dt
x_i' = x_i + v_i * cos(theta_mid) * dt
y_i' = y_i + v_i * sin(theta_mid) * dt
theta_i' = wrap(theta_i + omega_i * dt)
```

## Measurement Likelihood

For each tree observation and particle, the observation is projected into global coordinates:

```text
g_ij = [x_i + r_j * cos(theta_i + beta_j),
        y_i + r_j * sin(theta_i + beta_j)]
```

The nearest map landmark residual is:

```text
d_ij^2 = min_k ||g_ij - m_k||^2
```

Particle log weights accumulate a Gaussian residual model:

```text
log w_i <- log w_i - 0.5 * d_ij^2 / sigma^2
```

Weights are normalized after all observations.

## Coarse-To-Fine Scan-Match Reset

Wide orchard priors can be ambiguous because rows repeat. Before KLD resampling collapses the
particle cloud, TerraScout performs a deterministic scan-match reset around the current weighted
mean:

1. Search a coarse SE(2) lattice spanning `+/-5.5 m` and `+/-36 deg`.
2. Keep multiple spatially separated candidates instead of only the first local minimum.
3. Refine each candidate at two finer resolutions.
4. Score candidates by summed nearest-landmark residuals plus a small duplicate-match penalty.
5. Reset the particle cloud around the best pose with a tight Gaussian.

This keeps the method faithful to MCL: scan matching creates a better proposal distribution, and
the particle filter continues to carry uncertainty, prediction, likelihood weighting, and
KLD-adaptive resampling.

## KLD-Adaptive Resampling

When the effective sample size falls below 55 percent of the particle count:

```text
N_eff = 1 / sum_i(w_i^2)
```

particles are resampled with low-variance sampling. The target count is computed from occupied
pose bins using the standard KLD sampling approximation:

```text
N_required = (k - 1) / (2 epsilon) *
             (1 - 2/(9(k - 1)) + z * sqrt(2/(9(k - 1))))^3
```

The result is clamped between `min_particles` and `max_particles`.

## Pseudocode

```text
function relocalize(detections, map):
    candidates = coarse lattice scan match around prior mean
    candidates = fine refine top separated candidates
    best = argmin scan_match_score(candidate)
    reset particles around best

function update(detections, map):
    for each particle:
        for each tree observation:
            global_observation = transform observation by particle pose
            residual = distance to nearest map tree
            log_weight += gaussian_log_likelihood(residual)
    normalize weights
    if N_eff is low:
        KLD adaptive resample
```

## Acceptance Evidence

The localization benchmark initializes from a `+/-5 m`, `+/-30 degree` pose prior and reports
prior error, final pose error, and particle count. The current suite reaches p95 final pose error
below 0.03 m with no more than 3000 particles.

## References

- Thrun, Burgard, and Fox, Probabilistic Robotics, Monte Carlo Localization.
- Thrun et al., Robust Monte Carlo Localization for Mobile Robots.
