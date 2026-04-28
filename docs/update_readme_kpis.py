"""Update or verify the README KPI snapshot from reproduce_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "artifacts" / "reproduce_summary.json"
START = "<!-- TERRASCOUT_KPI_START -->"
END = "<!-- TERRASCOUT_KPI_END -->"


def build_kpi_block(summary_path: Path = SUMMARY) -> str:
    """Build the managed README KPI block."""

    summary = json.loads(summary_path.read_text())
    benchmark = summary["benchmark_summary"]
    mission = summary["mission"]
    return "\n".join(
        [
            START,
            "Layer KPI snapshot from the current reproducible benchmark suite:",
            "",
            "| Layer | Acceptance target | Current result |",
            "| --- | --- | ---: |",
            (
                "| L1 Kalman tracker | <0.20 m 1-second prediction error; >=95% association | "
                f"{benchmark['tracking_mean_prediction_error_m']:.3f} m mean; "
                f"{_percent(benchmark['tracking_mean_association_accuracy'])} association across 100 scenes |"
            ),
            (
                "| L2 particle filter | <0.15 m p95 pose error; <=3,000 particles | "
                f"{benchmark['localization_p95_pose_error_m']:.3f} m p95; "
                f"<={_localization_max_particles(summary_path)} particles across 10 wide-prior runs |"
            ),
            (
                "| L3 EKF-SLAM | <0.20 m pose error; <0.30 m landmark error | "
                f"{benchmark['slam_mean_pose_error_m']:.3f} m mean pose; "
                f"{benchmark['slam_mean_landmark_error_m']:.3f} m mean landmarks; "
                f"{benchmark['slam_mean_landmarks']:.0f} landmarks |"
            ),
            (
                "| L4 Hybrid A* | <=250 ms solve time; >=30% lower steering effort | "
                f"{benchmark['planner_mean_wall_time_ms']['hybrid_astar']:.1f} ms mean; "
                f"{benchmark['planner_mean_steering_reduction_percent']:.1f}% steering reduction |"
            ),
            (
                "| L5 MDP scheduler | <=5% oracle gap; <800 ms solve time | "
                f"{benchmark['scheduler_max_optimality_gap_percent']:.3f}% gap; "
                f"{_budget_status(benchmark['scheduler_max_wall_time_ms'], 800.0)} unconstrained; "
                f"{_budget_status(benchmark['resource_scheduler_max_wall_time_ms'], 800.0)} resource-aware |"
            ),
            (
                "| 30-row mission | >=9/10 priority goals; 0 collisions; <60 s wall time | "
                f"{int(benchmark['end_to_end_priority_goals'])}/10 goals; "
                f"{int(benchmark['end_to_end_total_collisions'])} collisions; "
                f"{benchmark['end_to_end_mean_pose_error_m']:.3f} m mean pose; "
                f"{_budget_status(benchmark['end_to_end_max_wall_time_s'], 60.0)} wall time |"
            ),
            (
                "| Default mission | 100% inspection success; no collisions | "
                f"{_percent(mission['success_rate'])} success; "
                f"{int(mission['collisions'])} collisions |"
            ),
            END,
        ]
    )


def update_readme(readme_path: Path = README, summary_path: Path = SUMMARY) -> str:
    """Return README text with the managed KPI block replaced."""

    text = readme_path.read_text()
    block = build_kpi_block(summary_path)
    start = text.index(START)
    end = text.index(END) + len(END)
    return f"{text[:start]}{block}{text[end:]}"


def _localization_max_particles(summary_path: Path) -> int:
    csv_path = summary_path.parent / "localization_benchmark.csv"
    if not csv_path.exists():
        return 3000
    lines = csv_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return 3000
    header = lines[0].split(",")
    particle_idx = header.index("particle_count")
    return max(int(row.split(",")[particle_idx]) for row in lines[1:])


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _budget_status(value: float, budget: float) -> str:
    return "budget met" if value <= budget else "budget missed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update README KPI table from reproduce summary.")
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--check", action="store_true", help="Fail if README KPI block is stale.")
    args = parser.parse_args()

    updated = update_readme(args.readme, args.summary)
    current = args.readme.read_text()
    if args.check:
        if updated != current:
            print("README KPI block is stale; run python docs/update_readme_kpis.py", file=sys.stderr)
            raise SystemExit(1)
        print("README KPI block is current")
        return
    args.readme.write_text(updated)
    print(args.readme)


if __name__ == "__main__":
    main()
