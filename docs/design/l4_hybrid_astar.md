# L4 Planning Design Note

## Purpose

TerraScout provides grid A* and a heading aware Hybrid A* search over a known orchard map. Their output is a checked geometric path for a differential drive follower. Final heading and bounded curvature are not guaranteed by the goal connector or the grid fallback.

Implementation: terrascout/plan/astar.py and terrascout/plan/hybrid_astar.py.

## Footprint and occupancy

The rover is modeled as a circle of radius 0.45 m. Trees have radius 0.18 m and workers normally have radius 0.35 m. Grid planning adds an explicit margin and the half diagonal of a cell to obstacle inflation:

```text
cell_size = 0.4 m
tree_centre_exclusion = 0.45 + 0.18 + 0.20 = 0.83 m
worker_centre_exclusion = 0.45 + 0.35 + 0.40 = 1.20 m
grid_padding = cell_size / sqrt(2)
boundary_margin = 0.45 + 0.10 = 0.55 m
```

The padding covers the space between grid cell centres. Continuous segment checks also validate the rover centre against obstacle circles and orchard boundaries. A physically clear endpoint may connect to a nearby free grid cell only when the entire connection is clear. An occupied goal is not silently replaced by an unreachable destination.

## Grid A*

The search uses eight neighbors with unit cardinal cost and diagonal cost 1.414. A diagonal step is rejected if either adjacent orthogonal cell is blocked. The heuristic is Euclidean distance in cell units. Reconstructed paths retain direction changes and include exact, checked start and goal connections. An unsuccessful search returns an empty path.

## Hybrid A*

Hybrid search uses a coarse position grid with 24 heading bins. Each state also retains a continuous pose. Forward and reverse motion primitives use a 0.8 m step and a nominal 1.2 m turn radius. Their integration matches the simulator's midpoint position update. Every intermediate segment is checked; testing just the endpoint would allow a primitive to cross an obstacle.

A lightweight connector tries forward and reverse approaches to the goal before and during lattice search. It is not an optimal Reeds Shepp solver. The final geometric goal connection and the grid fallback do not enforce the nominal turn radius or goal orientation. Only collinear points may be removed without a new collision check. Equal heading bins alone do not establish that the chord is clear.

Primitive cost includes distance, a turn penalty, and a reverse penalty. The heuristic adds a heading preference to Euclidean distance. Search is bounded by max_expansions; exhaustion invokes grid A*. Failure of that fallback also returns an empty path.

## Execution and evaluation

A missing path commands zero motion and triggers later retries. The follower turns before translating. Grid paths retain their sparse corners; lookahead is limited to dense Hybrid A* samples and requires additional clearance. A command guard checks proposed motion using the selected navigation estimate and known map. A separate evaluator uses simulated actual motion and physical body radii to report contact with trees, independently moving workers, or boundaries.

## Benchmark interpretation

Both planners report the same sum of geometric segment heading changes, plus path length, waypoint count, and solve time. This is not measured actuator steering effort. Negative steering reduction is retained when Hybrid A* performs worse. A zero waypoint result is a planning failure, not a successful short route. Read the current README snapshot and retained CI CSVs for measured results.

## References

* Dolgov, Thrun, Montemerlo, and Diebel, Path Planning for Autonomous Vehicles in Unknown Semi structured Environments.
* LaValle, Planning Algorithms, graph search and kinodynamic planning.
