# L5 MDP Scheduling Design Note

## Purpose

L5 chooses which orchard row goals to inspect under priority, travel, battery, and daylight
constraints. It is a compact finite-state MDP whose policy composes with the planner and mission
runner.

Implementation: `terrascout/scheduler/value_iteration.py`

Benchmark: `benchmarks/scheduler_benchmark.py`

## State

For the exact route-order problem:

```text
s = [visited_mask, current_position]
```

`visited_mask` is a bitmask of inspected goals. `current_position` is either a goal index or the
special start index.

For resource-aware planning, the recursive search also includes coarse battery and daylight bins:

```text
s_r = [visited_mask, current_position, battery_bin, time_bin]
```

## Actions

Each action selects one unvisited inspection goal:

```text
a in unvisited_goals
```

An action is feasible only if travel distance fits the remaining battery budget and travel plus
service time fits the daylight budget.

## Reward Model

The immediate reward is:

```text
R(s, a) = inspection_reward * priority[a] - travel_cost_per_m * travel_distance(s, a)
```

Dropped goals receive an implicit low-battery penalty in the resource-aware solver so the policy
prefers feasible inspections but does not invent impossible travel.

## Value Iteration

For the exact non-resource order, the Bellman backup is:

```text
V(mask, pos) = max_a R(mask, pos, a) + gamma * V(mask | (1 << a), a)
```

Iteration continues until the maximum value change is below tolerance or the maximum iteration
count is reached. The policy table stores the best action for each `(mask, pos)` state.

## Resource-Aware Pareto Search

The resource-aware planner keeps Pareto-optimal labels for each `(visited_mask, position)` state:

```text
label = [discounted_value_so_far, battery_remaining, time_remaining, order]
```

For each action:

```text
travel_m = distance(pos, action)
travel_s = travel_m / nominal_speed + service_time
```

If feasible, the planner creates a new label for the next mask, position, and exact remaining
resources. A candidate label is discarded when an existing label at the same state has greater or
equal value, battery, and time. Existing labels dominated by the candidate are removed. The final
plan is the label with the best discounted value plus terminal dropped-goal penalty. This keeps
the search exact for the benchmarked problem size without the optimality loss of coarse resource
bins.

## Pseudocode

```text
function plan_order(start, goals, priorities):
    initialize V and policy for all masks and positions
    repeat:
        for each state:
            policy[state] = argmax_a reward + discount * V[next_state]
        stop when value delta is small
    roll out policy from empty mask

function plan_with_resources(start, goals, priorities, battery, daylight):
    labels[(0, start)] = initial label
    while frontier is not empty:
        label = pop frontier
        for feasible unvisited action:
            next_label = advance resources and discounted value
            if next_label is not dominated:
                add it to labels and frontier
    return label with best value plus terminal dropped-goal penalty
```

## Acceptance Evidence

The scheduler benchmark compares the MDP order against a brute-force permutation oracle for a
deterministic 7-goal problem. The resource scheduler benchmark compares battery/daylight
constrained plans against an exact constrained oracle across 50 randomized 8-goal layouts. Current
tests require an optimality gap near zero and sub-second scheduler wall time.

## References

- Russell and Norvig, Artificial Intelligence: A Modern Approach, Markov decision processes.
- Sutton and Barto, Reinforcement Learning: An Introduction, value iteration.
