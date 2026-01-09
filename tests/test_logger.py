"""Tests for TelemetryLogger."""

import pytest
import tempfile
from pathlib import Path

from hitl_lab.core.logger import TelemetryLogger
from hitl_lab.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    EventType,
    RunMetadata,
    ToolResult,
)


class TestTelemetryLogger:
    """Tests for TelemetryLogger."""

    def test_in_memory_logger(self) -> None:
        """Test in-memory logger creation."""
        logger = TelemetryLogger(":memory:")
        assert logger is not None
        logger.close()

    def test_file_logger(self) -> None:
        """Test file-based logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            logger = TelemetryLogger(db_path)
            assert db_path.exists()
            logger.close()

    def test_log_run_lifecycle(self) -> None:
        """Test logging run start and end."""
        logger = TelemetryLogger(":memory:")

        metadata = RunMetadata(
            run_id="test-run",
            scenario_id="test_scenario",
            condition_id="baseline",
            seed=42,
        )

        logger.log_run_start(metadata)
        logger.log_run_end("test-run", success=True)

        # Verify we can retrieve the run
        retrieved = logger.get_run_metadata("test-run")

        assert retrieved is not None
        assert retrieved.run_id == "test-run"
        assert retrieved.task_success is True

        logger.close()

    def test_log_action_proposed(self) -> None:
        """Test logging proposed actions."""
        logger = TelemetryLogger(":memory:")

        metadata = RunMetadata(run_id="test-run")
        logger.log_run_start(metadata)

        action = Action(
            id="action-1",
            tool_name="test_tool",
            tool_args={"key": "value"},
        )

        logger.log_action_proposed("test-run", action)

        events = logger.get_events(
            run_id="test-run", event_type=EventType.ACTION_PROPOSED
        )

        assert len(events) == 1
        assert events[0].payload["action_id"] == "action-1"

        logger.close()

    def test_log_injected_error(self) -> None:
        """Test logging injected errors."""
        logger = TelemetryLogger(":memory:")

        metadata = RunMetadata(run_id="test-run")
        logger.log_run_start(metadata)

        action = Action(id="action-1", tool_name="test", tool_args={})
        logger.log_action_proposed("test-run", action, injected_error=True)

        events = logger.get_events(
            run_id="test-run", event_type=EventType.INJECTED_ERROR
        )

        assert len(events) == 1

        logger.close()

    def test_log_approval_flow(self) -> None:
        """Test logging approval request and decision."""
        logger = TelemetryLogger(":memory:")

        metadata = RunMetadata(run_id="test-run")
        logger.log_run_start(metadata)

        action = Action(id="action-1", tool_name="test", tool_args={})
        request = ApprovalRequest(
            run_id="test-run",
            action=action,
            policy_name="risk_based",
        )

        logger.log_approval_requested("test-run", request, "cli")

        decision = Decision(
            action_id="action-1",
            approved=True,
            latency_ms=500.0,
        )

        logger.log_approval_decided("test-run", decision, "cli")

        # Check events
        requested = logger.get_events(
            run_id="test-run", event_type=EventType.APPROVAL_REQUESTED
        )
        decided = logger.get_events(
            run_id="test-run", event_type=EventType.APPROVAL_DECIDED
        )

        assert len(requested) == 1
        assert len(decided) == 1
        assert decided[0].payload["latency_ms"] == 500.0

        logger.close()

    def test_get_all_runs(self) -> None:
        """Test retrieving all runs."""
        logger = TelemetryLogger(":memory:")

        for i in range(3):
            metadata = RunMetadata(
                run_id=f"run-{i}",
                scenario_id="test",
                condition_id=f"condition-{i}",
            )
            logger.log_run_start(metadata)
            logger.log_run_end(f"run-{i}", success=True)

        runs = logger.get_all_runs()

        assert len(runs) == 3

        logger.close()

    def test_event_filtering(self) -> None:
        """Test event filtering by type."""
        logger = TelemetryLogger(":memory:")

        metadata = RunMetadata(run_id="test-run")
        logger.log_run_start(metadata)

        action = Action(id="action-1", tool_name="test", tool_args={})
        logger.log_action_proposed("test-run", action)
        logger.log_tool_execution_start("test-run", action)

        # Filter by type
        action_events = logger.get_events(
            run_id="test-run", event_type=EventType.ACTION_PROPOSED
        )
        exec_events = logger.get_events(
            run_id="test-run", event_type=EventType.TOOL_EXECUTION_START
        )

        assert len(action_events) == 1
        assert len(exec_events) == 1

        logger.close()
