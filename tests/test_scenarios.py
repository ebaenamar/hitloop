"""Tests for scenarios."""

import pytest
import tempfile
from pathlib import Path

from hitloop.core.models import RiskClass
from hitloop.scenarios.email_draft import EmailDraftScenario
from hitloop.scenarios.record_update import RecordUpdateScenario


class TestEmailDraftScenario:
    """Tests for EmailDraftScenario."""

    def test_scenario_creation(self) -> None:
        """Test scenario creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            assert scenario.name == "email_draft"
            assert "send_email" in scenario.get_tools()

    def test_generate_correct_action(self) -> None:
        """Test generating correct action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            action = scenario.generate_action(correct=True)

            assert action.tool_name == "send_email"
            assert "recipient" in action.tool_args
            assert action.tool_args["recipient"] in scenario.VALID_RECIPIENTS

    def test_generate_incorrect_action(self) -> None:
        """Test generating incorrect action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            action = scenario.generate_action(correct=False)

            assert action.tool_args["recipient"] in scenario.INVALID_RECIPIENTS

    def test_send_email_tool(self) -> None:
        """Test email sending tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            tools = scenario.get_tools()

            result = tools["send_email"](
                recipient="test@example.com",
                subject="Test",
                body="Test body",
            )

            assert result["success"] is True
            assert "email_id" in result
            assert "file" in result

    def test_validate_success(self) -> None:
        """Test validation of successful result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            tools = scenario.get_tools()

            result = tools["send_email"](
                recipient="alice@example.com",
                subject="Test",
                body="Body",
            )

            validation = scenario.validate_result(result)
            assert validation.success is True

    def test_validate_invalid_recipient(self) -> None:
        """Test validation catches invalid recipient."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            tools = scenario.get_tools()

            result = tools["send_email"](
                recipient="hacker@evil.com",
                subject="Test",
                body="Body",
            )

            validation = scenario.validate_result(result)
            assert validation.success is False
            assert "invalid recipient" in validation.reason.lower()

    def test_reset(self) -> None:
        """Test scenario reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = EmailDraftScenario(output_dir=tmpdir)
            tools = scenario.get_tools()

            # Send an email
            tools["send_email"](
                recipient="test@example.com",
                subject="Test",
                body="Body",
            )

            # Reset
            scenario.reset()

            # Internal tracking should be cleared
            assert len(scenario._sent_emails) == 0


class TestRecordUpdateScenario:
    """Tests for RecordUpdateScenario."""

    def test_scenario_creation(self) -> None:
        """Test scenario creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            assert scenario.name == "record_update"
            assert "update_record" in scenario.get_tools()
            scenario.close()

    def test_get_record(self) -> None:
        """Test getting a record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            tools = scenario.get_tools()

            result = tools["get_record"](customer_id="CUST001")

            assert result["success"] is True
            assert "customer" in result
            assert result["customer"]["customer_id"] == "CUST001"

            scenario.close()

    def test_update_record(self) -> None:
        """Test updating a record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            tools = scenario.get_tools()

            result = tools["update_record"](
                customer_id="CUST001",
                field="email",
                value="newemail@example.com",
            )

            assert result["success"] is True
            assert result["new_value"] == "newemail@example.com"

            scenario.close()

    def test_update_invalid_customer(self) -> None:
        """Test updating non-existent customer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            tools = scenario.get_tools()

            result = tools["update_record"](
                customer_id="INVALID",
                field="email",
                value="test@example.com",
            )

            assert result["success"] is False
            assert "not found" in result["error"].lower()

            scenario.close()

    def test_validate_correct_update(self) -> None:
        """Test validation of correct update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            tools = scenario.get_tools()

            result = tools["update_record"](
                customer_id="CUST001",
                field="email",
                value="updated@example.com",
            )

            validation = scenario.validate_result(result)
            assert validation.success is True

            scenario.close()

    def test_validate_sensitive_field(self) -> None:
        """Test that sensitive field updates fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)
            tools = scenario.get_tools()

            result = tools["update_record"](
                customer_id="CUST001",
                field="credit_limit",
                value=99999.0,
            )

            validation = scenario.validate_result(result)
            assert validation.success is False
            assert "sensitive" in validation.reason.lower()

            scenario.close()

    def test_generate_action_respects_correctness(self) -> None:
        """Test action generation with correctness flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            scenario = RecordUpdateScenario(db_path=db_path)

            correct_action = scenario.generate_action(correct=True)
            assert correct_action.tool_args["customer_id"] in scenario.VALID_CUSTOMER_IDS
            assert correct_action.tool_args["field"] in scenario.VALID_FIELDS

            scenario.close()
