"""
HITL Lab - Human-in-the-Loop control library for AI agent workflows.

This library provides explicit control nodes for human oversight in LangGraph
agent workflows, with strong instrumentation for research experiments.
"""

from hitl_lab.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    RiskClass,
    TraceEvent,
    EventType,
)
from hitl_lab.core.interfaces import ApprovalBackend, HITLPolicy
from hitl_lab.core.logger import TelemetryLogger
from hitl_lab.backends.cli_backend import CLIBackend
from hitl_lab.policies.always_approve import AlwaysApprovePolicy
from hitl_lab.policies.risk_based import RiskBasedPolicy
from hitl_lab.policies.audit_plus_escalate import AuditPlusEscalatePolicy
from hitl_lab.langgraph.nodes import hitl_gate_node, execute_tool_node

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
