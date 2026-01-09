"""Tests for error injectors."""

import pytest

from hitl_lab.core.models import Action, RiskClass
from hitl_lab.eval.injectors import (
    ErrorInjector,
    InjectionConfig,
    InjectionType,
)


class TestInjectionConfig:
    """Tests for InjectionConfig."""

    def test_valid_config(self) -> None:
        """Test valid configuration."""
        config = InjectionConfig(injection_rate=0.2, seed=42)
        assert config.injection_rate == 0.2

    def test_invalid_rate_high(self) -> None:
        """Test that rate > 1.0 is rejected."""
        with pytest.raises(ValueError):
            InjectionConfig(injection_rate=1.5)

    def test_invalid_rate_negative(self) -> None:
        """Test that negative rate is rejected."""
        with pytest.raises(ValueError):
            InjectionConfig(injection_rate=-0.1)


class TestErrorInjector:
    """Tests for ErrorInjector."""

    def test_no_injection_at_zero_rate(self) -> None:
        """Test no injection when rate is 0."""
        injector = ErrorInjector(InjectionConfig(injection_rate=0.0))

        action = Action(tool_name="test", tool_args={"key": "value"})
        result = injector.maybe_inject(action)

        assert result.injected is False
        assert result.modified_action is None

    def test_always_inject_at_full_rate(self) -> None:
        """Test always inject when rate is 1.0."""
        injector = ErrorInjector(InjectionConfig(injection_rate=1.0, seed=42))

        action = Action(tool_name="send_email", tool_args={"recipient": "good@example.com"})
        result = injector.maybe_inject(action)

        assert result.injected is True
        assert result.modified_action is not None
        assert result.injection_type is not None

    def test_wrong_recipient_injection(self) -> None:
        """Test wrong recipient injection."""
        injector = ErrorInjector(
            InjectionConfig(
                injection_rate=1.0,
                injection_types=[InjectionType.WRONG_RECIPIENT],
                seed=42,
            )
        )

        action = Action(
            tool_name="send_email",
            tool_args={"recipient": "alice@example.com"},
        )
        result = injector.maybe_inject(action)

        assert result.injected is True
        assert result.injection_type == InjectionType.WRONG_RECIPIENT
        assert result.modified_action is not None
        assert result.modified_action.tool_args["recipient"] != "alice@example.com"
        assert "original_recipient" in result.injection_details

    def test_wrong_record_id_injection(self) -> None:
        """Test wrong record ID injection."""
        injector = ErrorInjector(
            InjectionConfig(
                injection_rate=1.0,
                injection_types=[InjectionType.WRONG_RECORD_ID],
                seed=42,
            )
        )

        action = Action(
            tool_name="update_record",
            tool_args={"customer_id": "CUST001", "field": "email"},
        )
        result = injector.maybe_inject(action)

        assert result.injected is True
        assert result.modified_action is not None
        assert result.modified_action.tool_args["customer_id"] != "CUST001"

    def test_wrong_amount_injection(self) -> None:
        """Test wrong amount injection."""
        injector = ErrorInjector(
            InjectionConfig(
                injection_rate=1.0,
                injection_types=[InjectionType.WRONG_AMOUNT],
                seed=42,
            )
        )

        action = Action(
            tool_name="transfer",
            tool_args={"amount": 100.0, "to": "account-123"},
        )
        result = injector.maybe_inject(action)

        assert result.injected is True
        assert result.modified_action is not None
        # Amount should be much larger (10x-100x)
        assert result.modified_action.tool_args["amount"] >= 1000.0

    def test_invalid_tool_injection(self) -> None:
        """Test invalid tool name injection."""
        injector = ErrorInjector(
            InjectionConfig(
                injection_rate=1.0,
                injection_types=[InjectionType.INVALID_TOOL],
                seed=42,
            )
        )

        action = Action(tool_name="normal_tool", tool_args={})
        result = injector.maybe_inject(action)

        assert result.injected is True
        assert result.modified_action is not None
        assert result.modified_action.tool_name != "normal_tool"
        assert result.modified_action.tool_name in ErrorInjector.INVALID_TOOLS

    def test_injected_action_marked_high_risk(self) -> None:
        """Test that injected actions are marked as HIGH risk."""
        injector = ErrorInjector(InjectionConfig(injection_rate=1.0, seed=42))

        action = Action(
            tool_name="test",
            tool_args={"recipient": "test@example.com"},
            risk_class=RiskClass.LOW,
        )
        result = injector.maybe_inject(action)

        assert result.modified_action is not None
        assert result.modified_action.risk_class == RiskClass.HIGH

    def test_stats_tracking(self) -> None:
        """Test injection statistics."""
        injector = ErrorInjector(InjectionConfig(injection_rate=0.5, seed=42))

        action = Action(tool_name="test", tool_args={"recipient": "x@y.com"})

        # Run several times
        for _ in range(100):
            injector.maybe_inject(action)

        stats = injector.get_stats()

        assert stats["total_actions"] == 100
        assert stats["configured_rate"] == 0.5
        # Actual rate should be approximately 0.5
        assert 0.3 < stats["actual_rate"] < 0.7

    def test_reset(self) -> None:
        """Test stats reset."""
        injector = ErrorInjector(InjectionConfig(injection_rate=0.5, seed=42))

        action = Action(tool_name="test", tool_args={})
        injector.maybe_inject(action)

        injector.reset()
        stats = injector.get_stats()

        assert stats["total_actions"] == 0
        assert stats["injected_count"] == 0
