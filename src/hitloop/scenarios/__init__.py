"""Scenario implementations for HITL Lab."""

from hitloop.scenarios.base import Scenario, ScenarioConfig, ValidationResult
from hitloop.scenarios.email_draft import EmailDraftScenario
from hitloop.scenarios.record_update import RecordUpdateScenario

__all__ = [
    "Scenario",
    "ScenarioConfig",
    "ValidationResult",
    "EmailDraftScenario",
    "RecordUpdateScenario",
]
