"""Tests for approval backends."""

import pytest

from hitl_lab.core.models import Action, ApprovalRequest
from hitl_lab.backends.cli_backend import (
    AutoApproveBackend,
    AutoRejectBackend,
    ScriptedBackend,
)


class TestAutoApproveBackend:
    """Tests for AutoApproveBackend."""

    @pytest.mark.asyncio
    async def test_auto_approves(self) -> None:
        """Test that backend auto-approves."""
        backend = AutoApproveBackend()

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        decision = await backend.request_approval(request)

        assert decision.approved is True
        assert decision.decided_by == "auto"

    @pytest.mark.asyncio
    async def test_simulated_delay(self) -> None:
        """Test simulated delay."""
        backend = AutoApproveBackend(delay_ms=100)

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        decision = await backend.request_approval(request)

        assert decision.latency_ms >= 100


class TestAutoRejectBackend:
    """Tests for AutoRejectBackend."""

    @pytest.mark.asyncio
    async def test_auto_rejects(self) -> None:
        """Test that backend auto-rejects."""
        backend = AutoRejectBackend()

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        decision = await backend.request_approval(request)

        assert decision.approved is False


class TestScriptedBackend:
    """Tests for ScriptedBackend."""

    @pytest.mark.asyncio
    async def test_follows_script(self) -> None:
        """Test that backend follows script."""
        backend = ScriptedBackend(decisions=[True, False, True])

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        d1 = await backend.request_approval(request)
        d2 = await backend.request_approval(request)
        d3 = await backend.request_approval(request)

        assert d1.approved is True
        assert d2.approved is False
        assert d3.approved is True

    @pytest.mark.asyncio
    async def test_uses_default_when_script_exhausted(self) -> None:
        """Test default value when script is exhausted."""
        backend = ScriptedBackend(decisions=[True], default_approve=False)

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        # First from script
        d1 = await backend.request_approval(request)
        assert d1.approved is True

        # Second uses default
        d2 = await backend.request_approval(request)
        assert d2.approved is False

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test script reset."""
        backend = ScriptedBackend(decisions=[True, False])

        action = Action(tool_name="test", tool_args={})
        request = ApprovalRequest(run_id="run-1", action=action)

        # Use script
        await backend.request_approval(request)
        await backend.request_approval(request)

        # Reset
        backend.reset()

        # Should start from beginning
        d = await backend.request_approval(request)
        assert d.approved is True
