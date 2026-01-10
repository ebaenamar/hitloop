"""
HITL Lab - Human-in-the-Loop control library for AI agent workflows.

This library provides explicit control nodes for human oversight in LangGraph
agent workflows, with strong instrumentation for research experiments.
"""

from hitloop.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    RiskClass,
    TraceEvent,
    EventType,
)
from hitloop.core.interfaces import ApprovalBackend, HITLPolicy
from hitloop.core.logger import TelemetryLogger
from hitloop.backends.cli_backend import CLIBackend
from hitloop.policies.always_approve import AlwaysApprovePolicy
from hitloop.policies.risk_based import RiskBasedPolicy
from hitloop.policies.audit_plus_escalate import AuditPlusEscalatePolicy
from hitloop.langgraph.nodes import hitl_gate_node, execute_tool_node

__version__ = "0.1.0"

__all__ = [
    # Models
    "Action",
    "ApprovalRequest",
    "Decision",
    "RiskClass",
    "TraceEvent",
    "EventType",
    # Interfaces
    "ApprovalBackend",
    "HITLPolicy",
    # Logger
    "TelemetryLogger",
    # Backends
    "CLIBackend",
    # Policies
    "AlwaysApprovePolicy",
    "RiskBasedPolicy",
    "AuditPlusEscalatePolicy",
    # LangGraph nodes
    "hitl_gate_node",
    "execute_tool_node",
]
