#!/usr/bin/env python3
"""Basic HITL workflow example.

This example demonstrates a simple HITL workflow using the email draft scenario
with the risk-based policy and CLI backend.

Usage:
    python examples/basic_workflow.py

For auto-approve mode (no prompts):
    python examples/basic_workflow.py --auto
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hitloop import (
    CLIBackend,
    RiskBasedPolicy,
    TelemetryLogger,
)
from hitloop.backends.cli_backend import AutoApproveBackend
from hitloop.core.models import ApprovalRequest, RunMetadata
from hitloop.scenarios.email_draft import EmailDraftScenario


async def run_workflow(auto_approve: bool = False) -> None:
    """Run a basic HITL workflow."""
    print("=" * 60)
    print("HITL Lab - Basic Workflow Example")
    print("=" * 60)
    print()

    # Initialize components
    logger = TelemetryLogger(":memory:")

    policy = RiskBasedPolicy(
        require_approval_for_high=True,
        require_approval_for_medium=True,
        high_risk_tools=["send_email"],
    )

    if auto_approve:
        print("[Auto-approve mode enabled]")
        backend = AutoApproveBackend(delay_ms=100)
    else:
        backend = CLIBackend()

    scenario = EmailDraftScenario(output_dir=Path("./example_output/emails"))

    # Create run
    run_id = "example-run-001"
    metadata = RunMetadata(
        run_id=run_id,
        scenario_id="email_draft",
        condition_id="basic_example",
    )
    logger.log_run_start(metadata)

    print(f"Run ID: {run_id}")
    print(f"Scenario: {scenario.name}")
    print(f"Policy: {policy.name}")
    print()

    # Generate an action (simulating LLM proposal)
    action = scenario.generate_action(correct=True)
    print(f"Proposed action: {action.summary()}")
    print()

    logger.log_action_proposed(run_id, action)

    # Check if approval needed
    needs_approval, reason = policy.should_request_approval(action, {"run_id": run_id})
    print(f"Approval needed: {needs_approval}")
    print(f"Reason: {reason}")
    print()

    approved = True
    if needs_approval:
        request = ApprovalRequest(
            run_id=run_id,
            action=action,
            summary_context=scenario.get_task_description(),
            policy_name=policy.name,
            policy_reason=reason,
        )

        logger.log_approval_requested(run_id, request, "cli")
        decision = await backend.request_approval(request)
        logger.log_approval_decided(run_id, decision, "cli")

        approved = decision.approved
        print()
        print(f"Decision: {'APPROVED' if approved else 'REJECTED'}")
        print(f"Reason: {decision.reason}")
        print(f"Latency: {decision.latency_ms:.1f}ms")
        print()

    if approved:
        # Execute the tool
        tools = scenario.get_tools()
        tool_func = tools[action.tool_name]

        print("Executing tool...")
        logger.log_tool_execution_start(run_id, action)

        result = tool_func(**action.tool_args)

        print(f"Result: {result}")
        print()

        # Validate
        validation = scenario.validate_result(result)
        print(f"Validation: {'SUCCESS' if validation.success else 'FAILED'}")
        print(f"Details: {validation.reason}")

        logger.log_run_end(run_id, validation.success, validation.details)
    else:
        print("Action was rejected. No execution.")
        logger.log_run_end(run_id, False, {"rejected": True})

    print()
    print("=" * 60)
    print("Workflow complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic HITL workflow example")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use auto-approve mode (no prompts)",
    )
    args = parser.parse_args()

    asyncio.run(run_workflow(auto_approve=args.auto))


if __name__ == "__main__":
    main()
