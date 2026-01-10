"""LangGraph interrupt-based HITL nodes.

This module provides HITL nodes that use LangGraph's native interrupt() function.
This is the recommended approach for production as it:
1. Uses LangGraph's built-in checkpointing for persistence
2. Is compatible with LangGraph Studio and agent-inbox
3. Follows LangGraph's standard patterns

The interrupt payload follows a standard format that can be consumed by
agent-inbox or any custom UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypedDict

from langgraph.types import interrupt, Command

from hitloop.core.interfaces import HITLPolicy
from hitloop.core.logger import TelemetryLogger
from hitloop.core.models import Action, Decision, ToolResult


class HITLInterruptPayload(TypedDict):
    """Standard payload format for HITL interrupts.
    
    This format is designed to be compatible with agent-inbox and
    other LangGraph ecosystem tools.
    """
    type: str  # "hitloop:approval_request"
    action_id: str
    tool_name: str
    tool_args: dict[str, Any]
    risk_class: str
    policy_name: str
    policy_reason: str
    summary: str
    context: str
    timestamp: str


class HITLResumePayload(TypedDict, total=False):
    """Expected format when resuming from an interrupt."""
    approved: bool
    reason: str
    decided_by: str
    tags: list[str]


def create_interrupt_gate_node(
    policy: HITLPolicy,
    logger: TelemetryLogger | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any] | Command]:
    """Create a HITL gate node using LangGraph's native interrupt().
    
    This node uses LangGraph's interrupt() function for human-in-the-loop,
    which integrates with LangGraph's checkpointing system. The graph will
    pause at this node and wait for a Command(resume=...) to continue.
    
    Args:
        policy: The HITL policy to determine when approval is needed
        logger: Optional telemetry logger for instrumentation
        
    Returns:
        A node function that can be added to a LangGraph StateGraph
        
    Example:
        >>> from langgraph.graph import StateGraph
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> 
        >>> policy = RiskBasedPolicy(high_risk_tools=["send_email"])
        >>> gate = create_interrupt_gate_node(policy)
        >>> 
        >>> builder = StateGraph(MyState)
        >>> builder.add_node("hitl_gate", gate)
        >>> graph = builder.compile(checkpointer=MemorySaver())
        >>> 
        >>> # Run until interrupt
        >>> result = graph.invoke({"proposed_action": action}, config)
        >>> # result["__interrupt__"] contains the approval request
        >>> 
        >>> # Resume with decision
        >>> graph.invoke(Command(resume={"approved": True}), config)
    """
    
    def gate_node(state: dict[str, Any]) -> dict[str, Any] | Command:
        run_id = state.get("run_id", "unknown")
        action = state.get("proposed_action")
        
        if action is None:
            return {
                "approval_decision": None,
                "hitl_status": "no_action",
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
            
            if logger:
                logger.log_approval_decided(run_id, decision, channel="auto")
            
            return {
                "approval_decision": decision,
                "hitl_status": "auto_approved",
            }
        
        # Log that approval is being requested
        if logger:
            from hitloop.core.models import ApprovalRequest
            request = ApprovalRequest(
                run_id=run_id,
                action=action,
                policy_name=policy.name,
                policy_reason=policy_reason,
            )
            logger.log_approval_requested(run_id, request, channel="interrupt")
        
        # Create interrupt payload (compatible with agent-inbox)
        interrupt_payload: HITLInterruptPayload = {
            "type": "hitloop:approval_request",
            "action_id": action.id,
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "risk_class": action.risk_class.value,
            "policy_name": policy.name,
            "policy_reason": policy_reason,
            "summary": action.summary(),
            "context": _build_summary_context(state),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Use LangGraph's native interrupt - this pauses the graph
        # and waits for Command(resume=...) to continue
        resume_value: HITLResumePayload = interrupt(interrupt_payload)
        
        # When resumed, resume_value contains the human's decision
        approved = resume_value.get("approved", False)
        reason = resume_value.get("reason", "Approved" if approved else "Rejected")
        decided_by = resume_value.get("decided_by", "human")
        
        decision = Decision(
            action_id=action.id,
            approved=approved,
            reason=reason,
            decided_by=decided_by,
            tags=resume_value.get("tags", []),
        )
        
        if logger:
            logger.log_approval_decided(run_id, decision, channel="interrupt")
        
        # Update state via policy hook
        updated_state = policy.post_decision_update(state.copy(), action, decision)
        
        return {
            **updated_state,
            "approval_decision": decision,
            "hitl_status": "human_decided",
        }
    
    return gate_node


def create_interrupt_tool_node(
    tool_registry: dict[str, Callable[..., Any]],
    logger: TelemetryLogger | None = None,
    require_approval: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a tool execution node for use with interrupt-based HITL.
    
    This node executes the approved tool and captures the result.
    It works with the interrupt gate node.
    
    Args:
        tool_registry: Dict mapping tool names to callable implementations
        logger: Optional telemetry logger
        require_approval: If True, only execute approved actions
        
    Returns:
        A node function for tool execution
    """
    
    def execute_node(state: dict[str, Any]) -> dict[str, Any]:
        run_id = state.get("run_id", "unknown")
        action = state.get("proposed_action")
        decision = state.get("approval_decision")
        
        if action is None:
            return {"tool_result": {"success": False, "error": "No action to execute"}}
        
        if isinstance(action, dict):
            action = Action(**action)
        
        # Check approval
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
            return {
                "tool_result": {
                    "success": False,
                    "error": f"Unknown tool: {action.tool_name}",
                    "action_id": action.id,
                }
            }
        
        # Execute
        if logger:
            logger.log_tool_execution_start(run_id, action)
        
        result = ToolResult(action_id=action.id, success=False)
        
        try:
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
                "action_id": result.action_id,
                "execution_time_ms": result.execution_time_ms(),
            }
        }
    
    return execute_node


def should_execute(state: dict[str, Any]) -> Literal["execute", "skip"]:
    """Conditional edge: route to execute if approved, skip otherwise."""
    decision = state.get("approval_decision")
    
    if decision is None:
        return "skip"
    
    if isinstance(decision, dict):
        approved = decision.get("approved", False)
    else:
        approved = getattr(decision, "approved", False)
    
    return "execute" if approved else "skip"


def _build_summary_context(state: dict[str, Any]) -> str:
    """Build context string from state for human review."""
    context_parts = []
    
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
    
    return ""
