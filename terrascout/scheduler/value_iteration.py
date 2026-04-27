"""Value-iteration scheduler for orchard lane inspection."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D


@dataclass(frozen=True)
class SchedulerConfig:
    """Reward model for row scheduling."""

    inspection_reward: float = 10.0
    travel_cost_per_m: float = 0.18
    discount: float = 0.97
    tolerance: float = 1e-5
    max_iterations: int = 400
    nominal_speed_mps: float = 1.0
    service_time_s: float = 2.0
    low_battery_penalty: float = 2.0


@dataclass(frozen=True)
class SchedulePlan:
    """Resource-aware inspection schedule."""

    order: list[int]
    expected_value: float
    battery_remaining_m: float
    time_remaining_s: float
    dropped_goals: int


@dataclass(frozen=True)
class _ResourceLabel:
    value: float
    battery_remaining_m: float
    time_remaining_s: float
    order: tuple[int, ...]


class InspectionScheduler:
    """Small finite-state MDP over visited-lane bitmasks."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self.iterations: int = 0

    def plan_order(
        self,
        start: Pose2D,
        goals: list[Point2D],
        priorities: list[float] | None = None,
    ) -> list[int]:
        """Return an inspection order that maximizes discounted row value."""

        if not goals:
            return []
        priorities = priorities or [1.0] * len(goals)
        values, policy = self._solve(start, goals, priorities)
        del values

        n = len(goals)
        all_seen = (1 << n) - 1
        mask = 0
        pos = n
        order: list[int] = []
        while mask != all_seen:
            action = int(policy[mask, pos])
            if action < 0 or action in order:
                break
            order.append(action)
            mask |= 1 << action
            pos = action
        return order

    def plan_with_resources(
        self,
        start: Pose2D,
        goals: list[Point2D],
        priorities: list[float] | None = None,
        battery_budget_m: float = 140.0,
        daylight_budget_s: float = 180.0,
    ) -> SchedulePlan:
        """Plan an inspection order while accounting for battery and daylight budgets."""

        if not goals:
            return SchedulePlan([], 0.0, battery_budget_m, daylight_budget_s, 0)
        priorities = priorities or [1.0] * len(goals)
        distances = self._distance_matrix(start, goals)
        n = len(goals)
        all_seen = (1 << n) - 1
        labels: dict[tuple[int, int], list[_ResourceLabel]] = {
            (0, n): [_ResourceLabel(0.0, battery_budget_m, daylight_budget_s, ())]
        }
        frontier: list[tuple[int, int, _ResourceLabel]] = [(0, n, labels[(0, n)][0])]
        expanded = 0

        while frontier:
            mask, pos, label = frontier.pop()
            expanded += 1
            if mask == all_seen:
                continue
            depth = len(label.order)
            for action in range(n):
                if mask & (1 << action):
                    continue
                travel_m = float(distances[pos, action])
                travel_s = travel_m / self.config.nominal_speed_mps + self.config.service_time_s
                if travel_m > label.battery_remaining_m or travel_s > label.time_remaining_s:
                    continue
                reward = (
                    self.config.inspection_reward * priorities[action]
                    - self.config.travel_cost_per_m * travel_m
                )
                next_label = _ResourceLabel(
                    value=label.value + (self.config.discount**depth) * reward,
                    battery_remaining_m=label.battery_remaining_m - travel_m,
                    time_remaining_s=label.time_remaining_s - travel_s,
                    order=(*label.order, action),
                )
                next_key = (mask | (1 << action), action)
                if self._add_pareto_label(labels.setdefault(next_key, []), next_label):
                    frontier.append((next_key[0], next_key[1], next_label))

        self.iterations = expanded
        best = _ResourceLabel(
            value=-self.config.low_battery_penalty * n,
            battery_remaining_m=battery_budget_m,
            time_remaining_s=daylight_budget_s,
            order=(),
        )
        best_total = best.value
        for state_labels in labels.values():
            for label in state_labels:
                remaining = n - len(label.order)
                terminal_value = (
                    label.value
                    + (self.config.discount ** len(label.order))
                    * (-self.config.low_battery_penalty * remaining)
                )
                if terminal_value > best_total:
                    best_total = terminal_value
                    best = label
        order = list(best.order)

        return SchedulePlan(
            order=order,
            expected_value=best_total,
            battery_remaining_m=max(0.0, best.battery_remaining_m),
            time_remaining_s=max(0.0, best.time_remaining_s),
            dropped_goals=n - len(order),
        )

    def _add_pareto_label(self, labels: list[_ResourceLabel], candidate: _ResourceLabel) -> bool:
        for label in labels:
            if (
                label.value >= candidate.value
                and label.battery_remaining_m >= candidate.battery_remaining_m
                and label.time_remaining_s >= candidate.time_remaining_s
            ):
                return False
        labels[:] = [
            label
            for label in labels
            if not (
                candidate.value >= label.value
                and candidate.battery_remaining_m >= label.battery_remaining_m
                and candidate.time_remaining_s >= label.time_remaining_s
            )
        ]
        labels.append(candidate)
        return True

    def _solve(
        self,
        start: Pose2D,
        goals: list[Point2D],
        priorities: list[float],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        n = len(goals)
        state_count = 1 << n
        values = np.zeros((state_count, n + 1), dtype=float)
        policy = np.full((state_count, n + 1), -1, dtype=np.int64)
        distances = self._distance_matrix(start, goals)
        all_seen = state_count - 1

        for iteration in range(self.config.max_iterations):
            delta = 0.0
            new_values = values.copy()
            for mask in range(state_count):
                if mask == all_seen:
                    continue
                for pos in range(n + 1):
                    best_value = -float("inf")
                    best_action = -1
                    for action in range(n):
                        if mask & (1 << action):
                            continue
                        next_mask = mask | (1 << action)
                        reward = (
                            self.config.inspection_reward * priorities[action]
                            - self.config.travel_cost_per_m * distances[pos, action]
                        )
                        candidate = reward + self.config.discount * values[next_mask, action]
                        if candidate > best_value:
                            best_value = candidate
                            best_action = action
                    new_values[mask, pos] = best_value
                    policy[mask, pos] = best_action
                    delta = max(delta, abs(best_value - values[mask, pos]))
            values = new_values
            self.iterations = iteration + 1
            if delta <= self.config.tolerance:
                break
        return values, policy

    def _distance_matrix(self, start: Pose2D, goals: list[Point2D]) -> NDArray[np.float64]:
        n = len(goals)
        distances = np.zeros((n + 1, n), dtype=float)
        for pos in range(n + 1):
            origin_x = start.x if pos == n else goals[pos].x
            origin_y = start.y if pos == n else goals[pos].y
            for goal_idx, goal in enumerate(goals):
                distances[pos, goal_idx] = hypot(goal.x - origin_x, goal.y - origin_y)
        return distances
