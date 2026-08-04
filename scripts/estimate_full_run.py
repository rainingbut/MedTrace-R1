"""Project full evaluation time and cost from a completed real pilot run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_run(
    metadata: dict[str, object],
    metrics: dict[str, object],
    benchmark_manifest: dict[str, object],
    hourly_rate: float | None,
    safety_factor: float,
) -> dict[str, object]:
    if metadata.get("dry_run"):
        raise ValueError("cost projection requires a real model pilot, not a dry-run")
    if safety_factor < 1:
        raise ValueError("safety factor must be at least 1")
    completed = int(metadata["completed_records"])
    elapsed_seconds = float(metadata["elapsed_seconds"])
    total_records = int(benchmark_manifest["combined_records"])
    if completed < 1 or elapsed_seconds <= 0:
        raise ValueError("pilot metadata has invalid completion or timing values")

    overall = metrics["overall"]
    projected_hours = elapsed_seconds / completed * total_records / 3600
    safe_hours = projected_hours * safety_factor
    report: dict[str, object] = {
        "pilot_records": completed,
        "pilot_elapsed_seconds": elapsed_seconds,
        "pilot_accuracy": overall["accuracy"],
        "pilot_parse_rate": overall["parse_rate"],
        "pilot_format_rate": overall["format_rate"],
        "pilot_truncation_rate": overall["truncation_rate"],
        "full_records": total_records,
        "projected_compute_hours": round(projected_hours, 3),
        "projected_hours_with_safety_factor": round(safe_hours, 3),
        "recommended_booking_hours": math.ceil(safe_hours + 0.5),
    }
    if hourly_rate is not None:
        if hourly_rate < 0:
            raise ValueError("--hourly-rate must not be negative")
        report["hourly_rate"] = hourly_rate
        report["projected_compute_cost"] = round(projected_hours * hourly_rate, 2)
        report["budget_with_safety_factor"] = round(safe_hours * hourly_rate, 2)
    return report


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    metadata = load_json(run_dir / "metadata.json")
    metrics = load_json(run_dir / "metrics.json")
    benchmark_manifest = load_json(REPO_ROOT / "data/benchmark/manifest.json")

    report = estimate_run(
        metadata,
        metrics,
        benchmark_manifest,
        args.hourly_rate,
        args.safety_factor,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
