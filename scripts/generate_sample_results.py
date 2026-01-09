#!/usr/bin/env python3
"""Generate sample experiment results.

This script runs a full experiment with N=20 trials per condition
and exports results to CSV and JSON.

Usage:
    python scripts/generate_sample_results.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hitl_lab import TelemetryLogger
from hitl_lab.eval.runner import ExperimentRunner, create_standard_conditions
from hitl_lab.scenarios.email_draft import EmailDraftScenario


async def main() -> None:
    """Generate sample results."""
    print("=" * 60)
    print("HITL Lab - Generating Sample Results")
    print("=" * 60)
    print()

    output_dir = Path(__file__).parent.parent / "sample_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    db_path = output_dir / "experiment.db"
    logger = TelemetryLogger(db_path)

    print(f"Database: {db_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Create scenario
    scenario = EmailDraftScenario(
        output_dir=output_dir / "emails",
    )

    # Create conditions with N=20 trials
    conditions = create_standard_conditions(
        scenario=scenario,
        n_trials=20,
        injection_rate=0.2,
        base_seed=42,
    )

    print(f"Conditions: {len(conditions)}")
    for c in conditions:
        print(f"  - {c.condition_id}: {c.policy.name}")
    print()

    # Create runner
    runner = ExperimentRunner(logger, output_dir=output_dir)
    for condition in conditions:
        runner.add_condition(condition)

    # Run all trials
    total_trials = sum(c.n_trials for c in conditions)
    print(f"Running {total_trials} total trials...")
    print()

    current = 0

    def progress(cur: int, total: int, msg: str) -> None:
        nonlocal current
        current = cur
        if cur % 10 == 0 or cur == total:
            pct = (cur / total) * 100
            print(f"[{pct:5.1f}%] {msg}")

    await runner.run_all(progress_callback=progress)

    print()
    print("Exporting results...")

    # Export
    csv_path, json_path = runner.export_results(
        csv_path="results.csv",
        json_path="summary.json",
    )

    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print()

    # Print summary
    summary = runner.get_summary()
    print("Summary:")
    print("-" * 40)

    for condition_id, stats in summary.get("conditions", {}).items():
        print(f"\n{condition_id}:")
        print(f"  Trials: {stats['n_trials']}")
        print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"  Approval rate: {stats['approval_rate'] * 100:.1f}%")
        print(f"  Injected errors: {stats['injected_errors']}")
        print(f"  Errors caught: {stats['errors_caught']}")
        if stats.get("error_catch_rate") is not None:
            print(f"  Error catch rate: {stats['error_catch_rate'] * 100:.1f}%")

    print()
    print("=" * 60)
    print("Sample results generation complete!")
    print(f"Results saved to: {output_dir}")

    logger.close()


if __name__ == "__main__":
    asyncio.run(main())
