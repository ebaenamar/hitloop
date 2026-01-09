"""Scenario implementations for HITL Lab."""

from hitl_lab.scenarios.base import Scenario, ScenarioConfig, ValidationResult
from hitl_lab.scenarios.email_draft import EmailDraftScenario
from hitl_lab.scenarios.record_update import RecordUpdateScenario

__all__ = [
    "Scenario",
    "ScenarioConfig",
    "ValidationResult",
    "EmailDraftScenario",
    "RecordUpdateScenario",
]
