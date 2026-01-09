"""Tests for HITL policies."""

import pytest

from hitl_lab.core.models import Action, Decision, RiskClass
from hitl_lab.policies.always_approve import AlwaysApprovePolicy
from hitl_lab.policies.risk_based import RiskBasedPolicy
from hitl_lab.policies.audit_plus_escalate import AuditPlusEscalatePolicy


class TestAlwaysApprovePolicy:
    """Tests for AlwaysApprovePolicy."""

    def test_never_requests_approval(self) -> None:
        """Test that policy never requests approval."""
        policy = AlwaysApprovePolicy()

        action = Action(
            tool_name="dangerous_tool",
            tool_args={"destroy": True},
            risk_class=RiskClass.HIGH,
        )

        needs_approval, reason = policy.should_request_approval(action, {})

        assert needs_approval is False
        assert "Tier 4" in reason or "auto-approved" in reason

    def test_policy_name(self) -> None:
        """Test policy name."""
        policy = AlwaysApprovePolicy()
        assert policy.name == "always_approve"


class TestRiskBasedPolicy:
    """Tests for RiskBasedPolicy."""

    def test_approves_low_risk_by_default(self) -> None:
        """Test that low risk actions don't need approval by default."""
        policy = RiskBasedPolicy()

        action = Action(
            tool_name="read_file",
            tool_args={"path": "/tmp/test"},
            risk_class=RiskClass.LOW,
        )

        needs_approval, _ = policy.should_request_approval(action, {})
        assert needs_approval is False

    def test_requests_approval_for_high_risk(self) -> None:
        """Test that high risk actions need approval."""
        policy = RiskBasedPolicy(require_approval_for_high=True)

        action = Action(
            tool_name="delete_all",
            tool_args={},
            risk_class=RiskClass.HIGH,
        )

        needs_approval, reason = policy.should_request_approval(action, {})

        assert needs_approval is True
        assert "HIGH" in reason

    def test_high_risk_tools_list(self) -> None:
        """Test high risk tools configuration."""
        policy = RiskBasedPolicy(high_risk_tools=["send_email", "delete"])

        action = Action(
            tool_name="send_email",
            tool_args={"to": "test@example.com"},
            risk_class=RiskClass.LOW,  # Even low risk
        )

        needs_approval, reason = policy.should_request_approval(action, {})

        assert needs_approval is True
        assert "high-risk tool list" in reason

    def test_sensitive_arg_patterns(self) -> None:
        """Test sensitive argument pattern matching."""
        policy = RiskBasedPolicy(
            sensitive_arg_patterns={"recipient": ["@external.com", "@competitor.com"]}
        )

        action = Action(
            tool_name="send_email",
            tool_args={"recipient": "spy@competitor.com"},
            risk_class=RiskClass.LOW,
        )

        needs_approval, reason = policy.should_request_approval(action, {})

        assert needs_approval is True
        assert "sensitive pattern" in reason

    def test_amount_threshold(self) -> None:
        """Test amount threshold checking."""
        policy = RiskBasedPolicy(max_amount_without_approval=1000.0)

        # Below threshold
        action1 = Action(
            tool_name="transfer",
            tool_args={"amount": 500},
        )
        needs1, _ = policy.should_request_approval(action1, {})
        assert needs1 is False

        # Above threshold
        action2 = Action(
            tool_name="transfer",
            tool_args={"amount": 5000},
        )
        needs2, reason = policy.should_request_approval(action2, {})
        assert needs2 is True
        assert "exceeds threshold" in reason

    def test_post_decision_tracks_rejections(self) -> None:
        """Test that rejections are tracked in state."""
        policy = RiskBasedPolicy()

        action = Action(tool_name="test", tool_args={})
        decision = Decision(action_id=action.id, approved=False, reason="Denied")

        state = {}
        updated = policy.post_decision_update(state, action, decision)

        assert "_risk_policy_rejections" in updated
        assert len(updated["_risk_policy_rejections"]) == 1


class TestAuditPlusEscalatePolicy:
    """Tests for AuditPlusEscalatePolicy."""

    def test_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises error."""
        with pytest.raises(ValueError):
            AuditPlusEscalatePolicy(audit_sample_rate=1.5)

    def test_escalates_high_risk(self) -> None:
        """Test escalation on high risk."""
        policy = AuditPlusEscalatePolicy(
            audit_sample_rate=0.0,  # No random audit
            escalate_on_high_risk=True,
        )

        action = Action(
            tool_name="dangerous",
            tool_args={},
            risk_class=RiskClass.HIGH,
        )

        needs_approval, reason = policy.should_request_approval(action, {})

        assert needs_approval is True
        assert "HIGH risk" in reason

    def test_escalates_on_anomaly_signals(self) -> None:
        """Test escalation on anomaly signals in state."""
        policy = AuditPlusEscalatePolicy(
            audit_sample_rate=0.0,
            anomaly_signals=["suspicious_activity"],
        )

        action = Action(tool_name="normal", tool_args={}, risk_class=RiskClass.LOW)
        state = {"anomaly_signals": ["suspicious_activity"]}

        needs_approval, reason = policy.should_request_approval(action, state)

        assert needs_approval is True
        assert "Anomaly" in reason

    def test_consecutive_action_threshold(self) -> None:
        """Test escalation on consecutive same-type actions."""
        policy = AuditPlusEscalatePolicy(
            audit_sample_rate=0.0,
            consecutive_action_threshold=3,
        )

        action = Action(tool_name="same_tool", tool_args={})

        # First two don't trigger
        needs1, _ = policy.should_request_approval(action, {})
        needs2, _ = policy.should_request_approval(action, {})

        # Third one should trigger
        needs3, reason = policy.should_request_approval(action, {})

        assert needs3 is True
        assert "consecutive" in reason

    def test_deterministic_audit_sampling(self) -> None:
        """Test that audit sampling is deterministic per action ID."""
        policy = AuditPlusEscalatePolicy(audit_sample_rate=0.5, seed=42)

        action = Action(
            id="fixed-id-for-test",
            tool_name="test",
            tool_args={},
            risk_class=RiskClass.LOW,
        )

        # Same action should always get same decision
        result1, _ = policy.should_request_approval(action, {})
        result2, _ = policy.should_request_approval(action, {})

        assert result1 == result2

    def test_reset_clears_history(self) -> None:
        """Test that reset clears action history."""
        policy = AuditPlusEscalatePolicy(
            audit_sample_rate=0.0,
            consecutive_action_threshold=3,
        )

        action = Action(tool_name="test", tool_args={})

        # Add some history
        policy.should_request_approval(action, {})
        policy.should_request_approval(action, {})

        # Reset
        policy.reset()

        # Should not trigger now
        needs, _ = policy.should_request_approval(action, {})
        assert needs is False
