"""Integration tests for HITL Lab.

These tests verify that all components work together correctly.
"""

import pytest
import tempfile
from pathlib import Path

from hitloop import (
    TelemetryLogger,
    RiskBasedPolicy,
)
from hitloop.backends.cli_backend import AutoApproveBackend, ScriptedBackend
from hitloop.core.models import Action, ApprovalRequest, RiskClass, RunMetadata
from hitloop.eval.runner import ExperimentRunner, ExperimentCondition
from hitloop.eval.injectors import InjectionConfig
from hitloop.eval.metrics import MetricsCalculator
from hitloop.scenarios.email_draft import EmailDraftScenario


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_basic_workflow_auto_approve(self) -> None:
        """Test a basic workflow with auto-approve backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            logger = TelemetryLogger(":memory:")
            policy = RiskBasedPolicy(require_approval_for_high=True)
            backend = AutoApproveBackend(delay_ms=10)
            scenario = EmailDraftScenario(output_dir=Path(tmpdir) / "emails")

            # Create run
            run_id = "integration-test-1"
            metadata = RunMetadata(
                run_id=run_id,
                scenario_id="email_draft",
                condition_id="test",
            )
            logger.log_run_start(metadata)

            # Generate action
            action = scenario.generate_action(correct=True)
            logger.log_action_proposed(run_id, action)

            # Check policy
            needs_approval, reason = policy.should_request_approval(action, {})

            if needs_approval:
                request = ApprovalRequest(
                    run_id=run_id,
                    action=action,
                    policy_name=policy.name,
                    policy_reason=reason,
                )
                logger.log_approval_requested(run_id, request, "test")
                decision = await backend.request_approval(request)
                logger.log_approval_decided(run_id, decision, "test")
                assert decision.approved is True

            # Execute
            tools = scenario.get_tools()
            result = tools[action.tool_name](**action.tool_args)

            # Validate
            validation = scenario.validate_result(result)
            logger.log_run_end(run_id, validation.success)

            # Verify logging worked
            events = logger.get_events(run_id=run_id)
            assert len(events) > 0

            run_metadata = logger.get_run_metadata(run_id)
            assert run_metadata is not None
            assert run_metadata.task_success is not None

            logger.close()

    @pytest.mark.asyncio
    async def test_rejection_workflow(self) -> None:
        """Test workflow with rejection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TelemetryLogger(":memory:")
            policy = RiskBasedPolicy(
                require_approval_for_high=True,
                require_approval_for_medium=True,
            )
            # Backend that rejects
            backend = ScriptedBackend(decisions=[False])
            scenario = EmailDraftScenario(output_dir=Path(tmpdir) / "emails")

            run_id = "rejection-test"
            metadata = RunMetadata(run_id=run_id)
            logger.log_run_start(metadata)

            # Generate high-risk action
            action = Action(
                tool_name="send_email",
                tool_args={"recipient": "test@example.com", "subject": "Test", "body": "Body"},
                risk_class=RiskClass.HIGH,
            )
            logger.log_action_proposed(run_id, action)

            needs_approval, reason = policy.should_request_approval(action, {})
            assert needs_approval is True

            request = ApprovalRequest(
                run_id=run_id,
                action=action,
                policy_name=policy.name,
            )

            decision = await backend.request_approval(request)
            logger.log_approval_decided(run_id, decision, "test")

            # Should be rejected
            assert decision.approved is False

            # Don't execute
            logger.log_run_end(run_id, False, {"rejected": True})

            run_metadata = logger.get_run_metadata(run_id)
            assert run_metadata.task_success is False

            logger.close()


class TestExperimentRunner:
    """Tests for experiment runner integration."""

    @pytest.mark.asyncio
    async def test_run_small_experiment(self) -> None:
        """Test running a small experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            logger = TelemetryLogger(output_dir / "experiment.db")
            scenario = EmailDraftScenario(output_dir=output_dir / "emails")

            # Single condition, few trials
            condition = ExperimentCondition(
                condition_id="test_condition",
                policy=RiskBasedPolicy(require_approval_for_medium=True),
                backend=AutoApproveBackend(delay_ms=5),
                scenario=scenario,
                injection_config=InjectionConfig(injection_rate=0.2, seed=42),
                n_trials=5,
                base_seed=42,
            )

            runner = ExperimentRunner(logger, output_dir=output_dir)
            runner.add_condition(condition)

            results = await runner.run_all()

            assert len(results) == 5

            # Check summary
            summary = runner.get_summary()
            assert "test_condition" in summary["conditions"]
            assert summary["conditions"]["test_condition"]["n_trials"] == 5

            logger.close()

    @pytest.mark.asyncio
    async def test_export_results(self) -> None:
        """Test exporting experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            logger = TelemetryLogger(output_dir / "experiment.db")
            scenario = EmailDraftScenario(output_dir=output_dir / "emails")

            condition = ExperimentCondition(
                condition_id="export_test",
                policy=RiskBasedPolicy(),
                backend=AutoApproveBackend(),
                scenario=scenario,
                n_trials=3,
            )

            runner = ExperimentRunner(logger, output_dir=output_dir)
            runner.add_condition(condition)

            await runner.run_all()

            # Export
            csv_path, json_path = runner.export_results(
                csv_path="results.csv",
                json_path="summary.json",
            )

            assert csv_path is not None
            assert csv_path.exists()
            assert json_path is not None
            assert json_path.exists()

            # Verify CSV content
            with open(csv_path) as f:
                lines = f.readlines()
                assert len(lines) == 4  # Header + 3 rows

            logger.close()


class TestMetricsIntegration:
    """Tests for metrics calculation integration."""

    @pytest.mark.asyncio
    async def test_compute_metrics_from_experiment(self) -> None:
        """Test computing metrics from experiment data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            logger = TelemetryLogger(output_dir / "metrics.db")
            scenario = EmailDraftScenario(output_dir=output_dir / "emails")

            condition = ExperimentCondition(
                condition_id="metrics_test",
                policy=RiskBasedPolicy(require_approval_for_medium=True),
                backend=AutoApproveBackend(),
                scenario=scenario,
                injection_config=InjectionConfig(injection_rate=0.3, seed=123),
                n_trials=10,
            )

            runner = ExperimentRunner(logger, output_dir=output_dir)
            runner.add_condition(condition)

            await runner.run_all()

            # Calculate metrics
            calculator = MetricsCalculator(logger)
            condition_metrics = calculator.compute_condition_metrics("metrics_test")

            assert condition_metrics.n_runs == 10
            assert 0.0 <= condition_metrics.success_rate <= 1.0
            assert 0.0 <= condition_metrics.approval_rate <= 1.0

            logger.close()
