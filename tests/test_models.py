"""Tests for core models."""

import pytest
from datetime import datetime, timezone

from hitl_lab.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    RiskClass,
    TraceEvent,
    EventType,
    ToolResult,
    RunMetadata,
)


class TestAction:
    """Tests for Action model."""

    def test_action_creation(self) -> None:
        """Test basic action creation."""
        action = Action(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
        )

        assert action.tool_name == "test_tool"
        assert action.tool_args == {"arg1": "value1"}
        assert action.risk_class == RiskClass.LOW
        assert action.id is not None

    def test_action_args_hash(self) -> None:
        """Test action args hash is deterministic."""
        action1 = Action(tool_name="tool", tool_args={"a": 1, "b": 2})
        action2 = Action(tool_name="tool", tool_args={"b": 2, "a": 1})

        # Same args should produce same hash regardless of order
        assert action1.args_hash() == action2.args_hash()

    def test_action_summary(self) -> None:
        """Test action summary generation."""
        action = Action(
            tool_name="send_email",
            tool_args={"recipient": "test@example.com"},
            risk_class=RiskClass.MEDIUM,
        )

        summary = action.summary()
        assert "MEDIUM" in summary
        assert "send_email" in summary
        assert "recipient" in summary


class TestDecision:
    """Tests for Decision model."""

    def test_decision_creation(self) -> None:
        """Test basic decision creation."""
        decision = Decision(
            action_id="test-action",
            approved=True,
            reason="Test approved",
            latency_ms=100.0,
        )

        assert decision.approved is True
        assert decision.latency_ms == 100.0

    def test_decision_latency_validation(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValueError):
            Decision(
                action_id="test",
                approved=True,
                latency_ms=-100.0,
            )


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_request_format_for_display(self) -> None:
        """Test approval request formatting."""
        action = Action(
            tool_name="dangerous_tool",
            tool_args={"target": "production"},
            risk_class=RiskClass.HIGH,
            rationale="Testing high risk action",
        )

        request = ApprovalRequest(
            run_id="run-123",
            action=action,
            summary_context="Test context",
            policy_name="risk_based",
            policy_reason="High risk tool",
        )

        formatted = request.format_for_display()

        assert "APPROVAL REQUEST" in formatted
        assert "dangerous_tool" in formatted
        assert "HIGH" in formatted
        assert "run-123" in formatted


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_trace_event_creation(self) -> None:
        """Test trace event creation."""
        event = TraceEvent(
            run_id="run-123",
            event_type=EventType.ACTION_PROPOSED,
            payload={"action_id": "action-1"},
        )

        assert event.run_id == "run-123"
        assert event.event_type == EventType.ACTION_PROPOSED
        assert event.timestamp is not None

    def test_trace_event_json_safe(self) -> None:
        """Test JSON-safe serialization."""
        event = TraceEvent(
            run_id="run-123",
            event_type=EventType.RUN_START,
            payload={"test": "data"},
        )

        data = event.model_dump_json_safe()

        assert isinstance(data["timestamp"], str)
        assert data["event_type"] == "run_start"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_tool_result_success(self) -> None:
        """Test successful tool result."""
        result = ToolResult(
            action_id="action-1",
            success=True,
            result={"data": "test"},
        )

        assert result.success is True
        assert result.error is None

    def test_tool_result_failure(self) -> None:
        """Test failed tool result."""
        result = ToolResult(
            action_id="action-1",
            success=False,
            error="Something went wrong",
            error_class="ValueError",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_tool_result_execution_time(self) -> None:
        """Test execution time calculation."""
        start = datetime.now(timezone.utc)
        result = ToolResult(
            action_id="action-1",
            success=True,
            started_at=start,
        )

        # No finished_at yet
        assert result.execution_time_ms() is None

        # Set finished_at
        result.finished_at = datetime.now(timezone.utc)
        exec_time = result.execution_time_ms()

        assert exec_time is not None
        assert exec_time >= 0


class TestRunMetadata:
    """Tests for RunMetadata model."""

    def test_run_metadata_creation(self) -> None:
        """Test run metadata creation."""
        metadata = RunMetadata(
            scenario_id="test_scenario",
            condition_id="baseline",
            seed=42,
        )

        assert metadata.scenario_id == "test_scenario"
        assert metadata.seed == 42
        assert metadata.run_id is not None

    def test_run_metadata_duration(self) -> None:
        """Test duration calculation."""
        metadata = RunMetadata(run_id="test")

        # No finished_at
        assert metadata.duration_ms() is None

        # Set finished_at
        metadata.finished_at = datetime.now(timezone.utc)
        duration = metadata.duration_ms()

        assert duration is not None
        assert duration >= 0
