"""Update or verify the README KPI snapshot from reproduce_summary.json."""

from __future__ import annotations

import argparse
import json
import hashlib
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
    return "\n".join([
        START,
        "Recorded simulation results. See the protocol and limitations below.",
        "",
        "| Experiment | Measured result |",
        "| --- | --- |",
        f"| Default mission ({mission.get('pose_source', 'unknown')}) | {_percent(mission['success_rate'])} requested goals serviced; {int(mission['collisions'])} contacts |",
        f"| Default suite, {int(benchmark.get('mission_runs', 0))} seeds | {_percent(benchmark.get('mission_mean_success_rate', 0.0))} mean requested goal completion; {int(benchmark.get('mission_total_collisions', 0))} contacts |",
        f"| 30 row priority trials | {int(benchmark.get('end_to_end_completed_runs', 0))}/{int(benchmark.get('end_to_end_runs', 0))} runs completed all goals; {int(benchmark.get('end_to_end_min_completed_goals', 0))}/{int(benchmark['end_to_end_priority_goals'])} minimum completed goals; {int(benchmark['end_to_end_total_collisions'])} contacts across {int(benchmark.get('end_to_end_runs', 0))} runs |",
        f"| Moving mission localization | {benchmark['end_to_end_mean_pose_error_m']:.3f} m mean selected pose error |",
        f"| Static relocalization, 10 poses | {benchmark['localization_p95_pose_error_m']:.3f} m p95 final position error |",
        f"| Constant velocity tracking, 100 scenes | {benchmark['tracking_mean_prediction_error_m']:.3f} m mean matched prediction error; {_percent(benchmark['tracking_mean_association_accuracy'])} ID continuity |",
        f"| Standalone EKF SLAM traversal | {benchmark['slam_mean_pose_error_m']:.3f} m mean final pose error; {benchmark['slam_mean_landmark_error_m']:.3f} m mean map error |",
        f"| Planner geometry comparison | {benchmark['planner_mean_steering_reduction_percent']:.1f}% Hybrid A* heading change reduction; {_budget_status(benchmark['planner_mean_wall_time_ms']['hybrid_astar'], 250.0)} solve time (mean <=250 ms) |",
        f"| Resource schedule versus exact oracle | {benchmark['resource_scheduler_max_optimality_gap_percent']:.3f}% maximum objective gap |",
        END,
    ])


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
        reference = json.loads(args.summary.read_text())
        expected = reference.get("provenance", {}).get("source_sha256")
        if expected:
            fingerprint = hashlib.sha256()
            for source in sorted((ROOT / "terrascout").rglob("*.py")):
                fingerprint.update(str(source.relative_to(ROOT)).encode())
                fingerprint.update(source.read_bytes())
            if fingerprint.hexdigest() != expected:
                print("Reference evidence is from different Python source; reproduce and refresh it", file=sys.stderr)
                raise SystemExit(1)
        if updated != current:
            print("README KPI block is stale; run python docs/update_readme_kpis.py", file=sys.stderr)
            raise SystemExit(1)
        print("README KPI block is current")
        return
    args.readme.write_text(updated)
    print(args.readme)


if __name__ == "__main__":
    main()
