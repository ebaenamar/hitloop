"""LangGraph node implementations for HITL Lab.

This module provides LangGraph-compatible nodes for HITL control flow:
- hitl_gate_node: Decision point for human approval
- execute_tool_node: Deterministic tool execution

These nodes can be composed into LangGraph workflows to create agent systems
with explicit human oversight at key decision points.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from hitl_lab.core.interfaces import ApprovalBackend, HITLPolicy
from hitl_lab.core.logger import TelemetryLogger
from hitl_lab.core.models import (
    Action,
    ApprovalRequest,
    Decision,
    RiskClass,
    ToolResult,
)


class HITLState(TypedDict, total=False):
    """Standard state structure for HITL-enabled LangGraph workflows.

    This TypedDict defines the expected state keys used by HITL nodes.
    Your workflow state should include these fields.

    Attributes:
        run_id: Unique identifier for the current run
        proposed_action: The action proposed by the LLM
        approval_decision: Result of human approval (if requested)
        tool_result: Result of tool execution
        messages: Conversation messages (LangGraph standard)
        trace: Optional in-memory trace events
    """

    run_id: str
    proposed_action: Action | None
    approval_decision: Decision | None
    tool_result: dict[str, Any] | None
    messages: list[Any]
    trace: list[dict[str, Any]]


def hitl_gate_node(
    policy: HITLPolicy,
    backend: ApprovalBackend,
    logger: TelemetryLogger | None = None,
    channel: str = "unknown",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a HITL gate node for LangGraph.

    The gate node is the central control point for human oversight. It:
    1. Extracts the proposed action from state
    2. Consults the policy to determine if approval is needed
    3. Requests approval from the backend if needed
    4. Updates state with the decision

    Args:
        policy: The HITL policy to use for approval decisions
        backend: The approval backend for human interaction
        logger: Optional telemetry logger for instrumentation
        channel: Identifier for the approval channel

    Returns:
        A callable node function compatible with LangGraph

    Example:
        >>> from langgraph.graph import StateGraph
        >>> policy = RiskBasedPolicy()
        >>> backend = CLIBackend()
        >>> logger = TelemetryLogger("trace.db")
        >>>
        >>> graph = StateGraph(HITLState)
        >>> graph.add_node("hitl_gate", hitl_gate_node(policy, backend, logger))
    """

    async def _async_gate(state: dict[str, Any]) -> dict[str, Any]:
        """Async implementation of the gate node."""
        run_id = state.get("run_id", "unknown")
        action = state.get("proposed_action")

        if action is None:
            # No action to approve
            return {
                "approval_decision": None,
                "tool_result": None,
            }

        # Ensure action is an Action object
        if isinstance(action, dict):
            action = Action(**action)

        # Log the proposed action
        if logger:
            injected = state.get("_injected_error", False)
            logger.log_action_proposed(run_id, action, injected_error=injected)

        # Consult policy
        needs_approval, policy_reason = policy.should_request_approval(action, state)

        if not needs_approval:
            # Auto-approve based on policy
            decision = Decision(
                action_id=action.id,
                approved=True,
                reason=policy_reason,
                decided_by=f"policy:{policy.name}",
                latency_ms=0.0,
            )
        else:
            # Request human approval
            request = ApprovalRequest(
                run_id=run_id,
                action=action,
                summary_context=_build_summary_context(state),
                policy_name=policy.name,
                policy_reason=policy_reason,
            )

            if logger:
                logger.log_approval_requested(run_id, request, channel)

            decision = await backend.request_approval(request)

            if logger:
                logger.log_approval_decided(run_id, decision, channel)

        # Update state via policy hook
        updated_state = policy.post_decision_update(state.copy(), action, decision)

        return {
            **updated_state,
            "approval_decision": decision,
        }

    def gate_node(state: dict[str, Any]) -> dict[str, Any]:
        """Sync wrapper for the async gate implementation."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a task
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(_async_gate(state))

    # Return async version for LangGraph (which supports async nodes)
    gate_node.__wrapped_async__ = _async_gate  # type: ignore
    return gate_node


def execute_tool_node(
    tool_registry: dict[str, Callable[..., Any]],
    logger: TelemetryLogger | None = None,
    require_approval: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a tool execution node for LangGraph.

    The execution node runs the approved tool and captures the result.
    It enforces that tools only execute after approval (when required).

    Args:
        tool_registry: Dict mapping tool names to callable implementations
        logger: Optional telemetry logger for instrumentation
        require_approval: If True, only execute approved actions

    Returns:
        A callable node function compatible with LangGraph

    Example:
        >>> tools = {
        ...     "send_email": send_email_tool,
        ...     "update_record": update_record_tool,
        ... }
        >>> executor = execute_tool_node(tools, logger)
        >>> graph.add_node("execute", executor)
    """

    async def _async_execute(state: dict[str, Any]) -> dict[str, Any]:
        """Async implementation of the execution node."""
        run_id = state.get("run_id", "unknown")
        action = state.get("proposed_action")
        decision = state.get("approval_decision")

        if action is None:
            return {"tool_result": {"success": False, "error": "No action to execute"}}

        # Ensure action is an Action object
        if isinstance(action, dict):
            action = Action(**action)

        # Check approval status
        if require_approval:
            if decision is None:
                return {
                    "tool_result": {
                        "success": False,
                        "error": "No approval decision found",
                        "action_id": action.id,
                    }
                }

            if isinstance(decision, dict):
                decision = Decision(**decision)

            if not decision.approved:
                return {
                    "tool_result": {
                        "success": False,
                        "error": "Action was rejected",
                        "action_id": action.id,
                        "rejection_reason": decision.reason,
                    }
                }

        # Look up tool
        tool_func = tool_registry.get(action.tool_name)
        if tool_func is None:
            error_msg = f"Unknown tool: {action.tool_name}"
            if logger:
                logger.log_error(run_id, ValueError(error_msg), {"action_id": action.id})
            return {
                "tool_result": {
                    "success": False,
                    "error": error_msg,
                    "action_id": action.id,
                }
            }

        # Execute tool
        if logger:
            logger.log_tool_execution_start(run_id, action)

        result = ToolResult(action_id=action.id, success=False)

        try:
            # Support both sync and async tools
            if asyncio.iscoroutinefunction(tool_func):
                output = await tool_func(**action.tool_args)
            else:
                output = tool_func(**action.tool_args)

            result.success = True
            result.result = output

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.error_class = type(e).__name__

            if logger:
                logger.log_error(run_id, e, {"action_id": action.id})

        finally:
            result.finished_at = datetime.now(timezone.utc)

        if logger:
            logger.log_tool_execution_end(run_id, result)

        return {
            "tool_result": {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "error_class": result.error_class,
                "action_id": result.action_id,
                "execution_time_ms": result.execution_time_ms(),
            }
        }

    def execute_node(state: dict[str, Any]) -> dict[str, Any]:
        """Sync wrapper for the async execution implementation."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(_async_execute(state))

    execute_node.__wrapped_async__ = _async_execute  # type: ignore
    return execute_node


def _build_summary_context(state: dict[str, Any]) -> str:
    """Build a summary context string from state for human review.

    Args:
        state: Current workflow state

    Returns:
        Human-readable summary of relevant context
    """
    context_parts = []

    # Include recent messages if available
    messages = state.get("messages", [])
    if messages:
        recent = messages[-3:] if len(messages) > 3 else messages
        for msg in recent:
            if hasattr(msg, "content"):
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                context_parts.append(f"- {content}")
            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                context_parts.append(f"- {content}")

    if context_parts:
        return "Recent conversation:\n" + "\n".join(context_parts)

    return "No additional context available"


def create_hitl_workflow_nodes(
    policy: HITLPolicy,
    backend: ApprovalBackend,
    tool_registry: dict[str, Callable[..., Any]],
    logger: TelemetryLogger | None = None,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Create both HITL nodes with shared configuration.

    Convenience function to create both the gate and execution nodes
    with consistent configuration.

    Args:
        policy: The HITL policy
        backend: The approval backend
        tool_registry: Available tools
        logger: Optional telemetry logger

    Returns:
        Dict with "hitl_gate" and "execute_tool" node functions

    Example:
        >>> nodes = create_hitl_workflow_nodes(policy, backend, tools, logger)
        >>> graph.add_node("hitl_gate", nodes["hitl_gate"])
        >>> graph.add_node("execute_tool", nodes["execute_tool"])
    """
    return {
        "hitl_gate": hitl_gate_node(policy, backend, logger),
        "execute_tool": execute_tool_node(tool_registry, logger),
    }


def should_execute_condition(state: dict[str, Any]) -> str:
    """Conditional edge function for deciding execution path.

    Use this as a conditional edge to route to execution or skip.

    Args:
        state: Current workflow state

    Returns:
        "execute" if approved, "skip" otherwise

    Example:
        >>> graph.add_conditional_edges(
        ...     "hitl_gate",
        ...     should_execute_condition,
        ...     {"execute": "execute_tool", "skip": "end"}
        ... )
    """
    decision = state.get("approval_decision")

    if decision is None:
        return "skip"

    if isinstance(decision, dict):
        approved = decision.get("approved", False)
    else:
        approved = decision.approved

    return "execute" if approved else "skip"
