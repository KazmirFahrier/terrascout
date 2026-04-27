# L4 Hybrid A* Planning Design Note

## Purpose

L4 converts row goals into kinematically feasible pose paths while avoiding tree trunks and
predicted worker positions. TerraScout keeps both a grid A* baseline and a heading-aware Hybrid
A* planner so benchmark results can compare smoothness and steering effort.

Implementation: `terrascout/plan/astar.py`, `terrascout/plan/hybrid_astar.py`

Benchmark: `benchmarks/planner_benchmark.py`

## Occupancy Model

The planner builds an inflated grid over the orchard:

```text
cell_size = 0.4 m
tree_radius = 0.38 m
worker_radius = 0.75 m
```

Tree landmarks and Kalman-predicted worker positions mark occupied cells within their inflation
radii. Start and goal cells are projected to the nearest free cell when necessary.

## Grid A* Baseline

The baseline planner uses an 8-connected grid:

```text
N = {left, right, up, down, diagonals}
cost = 1 for cardinal moves
cost = 1.414 for diagonal moves
heuristic = Euclidean distance in cells
```

The reconstructed path is sparsified by retaining cells at direction changes.

## Hybrid A* State

Hybrid A* searches a coarse lattice:

```text
s = [cell_x, cell_y, heading_bin]
heading_bins = 24
```

Each state also stores a continuous pose for expansion:

```text
p = [x, y, theta]
```

## Motion Primitives

Each expansion applies forward and reverse arc primitives:

```text
direction in {+1, -1}
turn in {-step / r_min, 0, +step / r_min}
theta_mid = theta + 0.5 * direction * turn
x' = x + direction * step * cos(theta_mid)
y' = y + direction * step * sin(theta_mid)
theta' = wrap(theta + direction * turn)
```

The implementation uses a minimum-turn-radius lattice (`r_min = 1.2 m`), reverse penalties, and
a lightweight analytic connector. The connector simulates bounded-curvature forward and reverse
approaches to the goal and accepts the connector only when all sampled poses remain in free
space. This is not a full optimal Reeds-Shepp solver, but it provides the same Hybrid A*
analytic-expansion role: quickly finish near-goal states with feasible curvature before falling
back to lattice expansion.

## Cost And Heuristic

Primitive cost:

```text
cost = step
if turning: cost += turn_cost
if reversing: cost += reverse_cost
```

Heuristic:

```text
h = EuclideanDistance(p, goal) + heading_cost * abs(wrap(heading_to_goal - theta))
```

The planner first tries the analytic connector directly from the start. During lattice search, it
tries the connector again whenever a state is within `analytic_expansion_distance_m` of the goal.
If the search exceeds `max_expansions`, it falls back to the grid A* baseline and annotates
heading from path segments.

## Pseudocode

```text
function hybrid_plan(start, goal, predicted_workers):
    blocked = inflated occupancy grid
    if analytic_connector(start, goal) is collision-free:
        return connector path
    open = priority queue with start
    while open is not empty and expansions < max_expansions:
        current = pop best f = g + h
        if current reaches goal:
            return reconstruct path
        if current is near goal and analytic_connector(current, goal) is collision-free:
            return reconstruct prefix + connector
        for primitive in forward/reverse arc primitives:
            next = integrate primitive
            if next cell is free and tentative cost improves:
                record parent and push
    return grid_astar_fallback(start, goal)
```

## Acceptance Evidence

The planner benchmark compares grid A* and Hybrid A* on deterministic orchard scenes and reports
waypoint count, path length, steering effort, and wall time. Current results show the Hybrid A*
path has much lower steering effort than the grid baseline while the analytic connector keeps
planning time comfortably within the CPU budget.

## References

- Dolgov, Thrun, Montemerlo, and Diebel, Path Planning for Autonomous Vehicles in Unknown
  Semi-structured Environments.
- LaValle, Planning Algorithms, graph search and kinodynamic planning.
