"""LangGraph integration for HITL Lab."""

from hitloop.langgraph.nodes import (
    HITLState,
    hitl_gate_node,
    execute_tool_node,
    create_hitl_workflow_nodes,
    should_execute_condition,
)
from hitloop.langgraph.interrupt_nodes import (
    HITLInterruptPayload,
    HITLResumePayload,
    create_interrupt_gate_node,
    create_interrupt_tool_node,
    should_execute,
)

__all__ = [
    # Legacy nodes (backend-based)
    "HITLState",
    "hitl_gate_node",
    "execute_tool_node",
    "create_hitl_workflow_nodes",
    "should_execute_condition",
    # Interrupt-based nodes (recommended)
    "HITLInterruptPayload",
    "HITLResumePayload",
    "create_interrupt_gate_node",
    "create_interrupt_tool_node",
    "should_execute",
]
