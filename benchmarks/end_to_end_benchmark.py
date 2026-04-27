"""30-row end-to-end acceptance benchmark."""

from __future__ import annotations

from pathlib import Path

from terrascout.eval.benchmarks import EndToEndBenchmarkRow, run_end_to_end_benchmark


def run(output: Path = Path("artifacts/end_to_end_benchmark.csv")) -> list[EndToEndBenchmarkRow]:
    return run_end_to_end_benchmark(output)


if __name__ == "__main__":
    rows = run()
    mean_success = sum(row.success_rate for row in rows) / len(rows)
    total_collisions = sum(row.collisions for row in rows)
    mean_wall_time = sum(row.wall_time_s for row in rows) / len(rows)
    print(
        f"end-to-end seeds={len(rows)} "
        f"mean_success={mean_success:.3f} "
        f"collisions={total_collisions} "
        f"mean_wall_s={mean_wall_time:.2f}"
    )
