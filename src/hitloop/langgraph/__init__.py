"""LangGraph integration for HITL Lab."""

from hitloop.langgraph.nodes import hitl_gate_node, execute_tool_node
from hitloop.langgraph.interrupt_nodes import (
    # agent-inbox compatible types
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
    ActionRequest,
    InterruptCallback,
    # Legacy aliases
    HITLInterruptPayload,
    HITLResumePayload,
    # Node factories
    create_interrupt_gate_node,
    create_interrupt_tool_node,
    should_execute,
)

__all__ = [
    "hitl_gate_node",
    "execute_tool_node",
    "HumanInterrupt",
    "HumanInterruptConfig",
    "HumanResponse",
    "ActionRequest",
    "InterruptCallback",
    "HITLInterruptPayload",
    "HITLResumePayload",
    "create_interrupt_gate_node",
    "create_interrupt_tool_node",
    "should_execute",
]
