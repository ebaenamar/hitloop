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
from functools import wraps
from typing import Any, Callable, Literal, TypedDict, Union

from langgraph.types import interrupt, Command

from hitloop.core.interfaces import HITLPolicy
from hitloop.core.logger import TelemetryLogger
from hitloop.core.models import Action, Decision, ToolResult, RiskClass

# Type for risk_class: can be a fixed value or a function that computes it from args
RiskClassResolver = Union[RiskClass, Callable[[dict[str, Any]], RiskClass]]
# Type for anomaly validators: functions that return (is_anomaly: bool, reason: str)
AnomalyValidator = Callable[[dict[str, Any]], tuple[bool, str]]


class HumanInterruptConfig(TypedDict):
    """Configuration for what actions are allowed on an interrupt.
    
    This is the exact format expected by agent-inbox.
    """
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    """Action request format for agent-inbox."""
    action: str
    args: dict[str, Any]


class HumanInterrupt(TypedDict, total=False):
    """Interrupt payload format compatible with agent-inbox.
    
    This is the exact schema expected by LangChain's agent-inbox UI.
    See: https://github.com/langchain-ai/agent-inbox
    """
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str  # Markdown description for the UI


class HumanResponse(TypedDict, total=False):
    """Response format from agent-inbox.
    
    The agent-inbox always returns a list with a single HumanResponse.
    """
    type: Literal["accept", "ignore", "response", "edit"]
    args: Any  # None, str, or ActionRequest depending on type


# Legacy aliases for backwards compatibility
HITLInterruptPayload = HumanInterrupt
HITLResumePayload = HumanResponse


# Type alias for interrupt callback
InterruptCallback = Callable[[HumanInterrupt, str], Any]  # (payload, thread_id) -> Any


def create_interrupt_gate_node(
    policy: HITLPolicy,
    logger: TelemetryLogger | None = None,
    on_interrupt: InterruptCallback | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any] | Command]:
    """Create a HITL gate node using LangGraph's native interrupt().
    
    This node uses LangGraph's interrupt() function for human-in-the-loop,
    which integrates with LangGraph's checkpointing system. The graph will
    pause at this node and wait for a Command(resume=...) to continue.
    
    Args:
        policy: The HITL policy to determine when approval is needed
        logger: Optional telemetry logger for instrumentation
        on_interrupt: Optional callback called when an interrupt is created.
            Receives (payload, thread_id) - useful for sending webhooks/push notifications.
        
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
        # See: https://github.com/langchain-ai/agent-inbox
        description = _build_interrupt_description(action, policy_reason, state)
        
        interrupt_payload: HumanInterrupt = {
            "action_request": {
                "action": action.tool_name,
                "args": action.tool_args,
            },
            "config": {
                "allow_ignore": True,   # Can skip/reject
                "allow_respond": True,  # Can add a comment
                "allow_edit": True,     # Can modify args
                "allow_accept": True,   # Can approve as-is
            },
            "description": description,
        }
        
        # Call webhook/notification callback if provided
        if on_interrupt:
            thread_id = state.get("configurable", {}).get("thread_id", run_id)
            try:
                result = on_interrupt(interrupt_payload, thread_id)
                # Handle async callbacks
                import asyncio
                if asyncio.iscoroutine(result):
                    asyncio.get_event_loop().run_until_complete(result)
            except Exception as e:
                # Don't fail the interrupt if webhook fails
                if logger:
                    logger.logger.warning(f"on_interrupt callback failed: {e}")
        
        # Use LangGraph's native interrupt - MUST pass a list like the official example
        # See: https://github.com/langchain-ai/agent-inbox-langgraph-example
        # agent-inbox expects interrupt([request]) and returns the response
        resume_value = interrupt([interrupt_payload])
        
        # The resume value can be:
        # - A list with one HumanResponse (from agent-inbox)
        # - A single HumanResponse dict (from manual Command(resume=...))
        if isinstance(resume_value, list) and len(resume_value) > 0:
            response: HumanResponse = resume_value[0]
        elif isinstance(resume_value, dict):
            response = resume_value
        else:
            response = {"type": "ignore", "args": None}
        
        # Parse the response based on type
        response_type = response.get("type", "ignore")
        response_args = response.get("args")
        
        if response_type == "accept":
            approved = True
            reason = "Accepted by human"
            decided_by = "human:accept"
        elif response_type == "edit":
            approved = True
            reason = "Accepted with edits"
            decided_by = "human:edit"
            # Update action args if edited
            if isinstance(response_args, dict) and "args" in response_args:
                action = Action(
                    tool_name=response_args.get("action", action.tool_name),
                    tool_args=response_args.get("args", action.tool_args),
                    risk_class=action.risk_class,
                    rationale=action.rationale,
                )
        elif response_type == "response":
            # Human provided a text response - treat as rejection with reason
            approved = False
            reason = str(response_args) if response_args else "Rejected with response"
            decided_by = "human:response"
        else:  # ignore
            approved = False
            reason = "Ignored by human"
            decided_by = "human:ignore"
        
        decision = Decision(
            action_id=action.id,
            approved=approved,
            reason=reason,
            decided_by=decided_by,
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


def _build_interrupt_description(
    action: Action,
    policy_reason: str,
    state: dict[str, Any],
) -> str:
    """Build a markdown description for agent-inbox.
    
    This is rendered in the agent-inbox UI as the main description
    of the interrupt event.
    """
    import json
    
    args_str = json.dumps(action.tool_args, indent=2)
    context = _build_summary_context(state)
    
    description = f"""## Approval Required

**Tool:** `{action.tool_name}`  
**Risk Level:** {action.risk_class.value.upper()}

### Why approval is needed
{policy_reason}

### Arguments
```json
{args_str}
```
"""
    
    if action.rationale:
        description += f"\n### Rationale\n{action.rationale}\n"
    
    if context:
        description += f"\n### Context\n{context}\n"
    
    return description


def hitl_action(
    risk_class: RiskClassResolver = RiskClass.MEDIUM,
    anomaly_validators: list[AnomalyValidator] | None = None,
    description: str | None = None,
    on_interrupt: InterruptCallback | None = None,
) -> Callable:
    """Decorator to add HITL approval to any tool function.
    
    This decorator wraps a tool function to require human approval before execution.
    It uses LangGraph's native interrupt() for the approval flow.
    
    Args:
        risk_class: Risk level for this tool. Can be:
            - A fixed RiskClass value (e.g., RiskClass.HIGH)
            - A function that takes args dict and returns RiskClass
              (for dynamic risk based on arguments)
        anomaly_validators: Optional list of functions to detect anomalies.
            Each validator receives the args dict and returns (is_anomaly, reason).
            If any validator returns is_anomaly=True, risk is escalated to HIGH.
        description: Custom description template for approval UI.
            If None, a default description is generated.
        on_interrupt: Optional callback for webhooks/notifications.
        
    Returns:
        Decorated function that will interrupt for approval before executing.
        
    Example:
        >>> from hitloop.langgraph import hitl_action
        >>> from hitloop import RiskClass
        >>> 
        >>> @hitl_action(risk_class=RiskClass.HIGH)
        ... def send_email(to: str, subject: str, body: str) -> dict:
        ...     return {"status": "sent", "to": to}
        
    Example with dynamic risk:
        >>> def email_risk(args: dict) -> RiskClass:
        ...     if "all@" in args.get("to", ""):
        ...         return RiskClass.HIGH  # Broadcast email
        ...     return RiskClass.MEDIUM
        >>> 
        >>> @hitl_action(risk_class=email_risk)
        ... def send_email(to: str, subject: str) -> dict:
        ...     return {"status": "sent"}
        
    Example with anomaly detection:
        >>> def check_recipient(args: dict) -> tuple[bool, str]:
        ...     blocked = ["competitor.com", "spam.com"]
        ...     to = args.get("to", "")
        ...     for domain in blocked:
        ...         if domain in to:
        ...             return True, f"Blocked domain: {domain}"
        ...     return False, ""
        >>> 
        >>> @hitl_action(anomaly_validators=[check_recipient])
        ... def send_email(to: str, subject: str) -> dict:
        ...     return {"status": "sent"}
    """
    validators = anomaly_validators or []
    
    def decorator(func: Callable) -> Callable:
        tool_name = func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import json
            import inspect
            
            # Build args dict from positional and keyword args
            action_args = kwargs.copy()
            if args:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(params):
                        action_args[params[i]] = arg
            
            # Resolve risk class (can be value or function)
            if callable(risk_class):
                resolved_risk = risk_class(action_args)
            else:
                resolved_risk = risk_class
            
            # Run anomaly validators
            anomaly_reasons = []
            for validator in validators:
                try:
                    is_anomaly, reason = validator(action_args)
                    if is_anomaly:
                        anomaly_reasons.append(reason)
                        resolved_risk = RiskClass.HIGH  # Escalate on anomaly
                except Exception as e:
                    anomaly_reasons.append(f"Validator error: {e}")
                    resolved_risk = RiskClass.HIGH
            
            # Build description
            args_str = json.dumps(action_args, indent=2, default=str)
            if description:
                desc = description.format(tool_name=tool_name, args=action_args, risk=resolved_risk.value)
            else:
                desc = f"## Approval Required\n\n**Tool:** `{tool_name}`\n**Risk Level:** {resolved_risk.value.upper()}\n\n"
                if anomaly_reasons:
                    desc += "### ⚠️ Anomalies Detected\n" + "\n".join(f"- {r}" for r in anomaly_reasons) + "\n\n"
                desc += f"### Arguments\n```json\n{args_str}\n```"
            
            # Create interrupt payload
            interrupt_payload: HumanInterrupt = {
                "action_request": {
                    "action": tool_name,
                    "args": action_args,
                },
                "config": {
                    "allow_ignore": True,
                    "allow_respond": True,
                    "allow_edit": True,
                    "allow_accept": True,
                },
                "description": desc,
            }
            
            # Call webhook callback if provided
            if on_interrupt:
                try:
                    import asyncio
                    result = on_interrupt(interrupt_payload, "unknown")
                    if asyncio.iscoroutine(result):
                        asyncio.get_event_loop().run_until_complete(result)
                except Exception:
                    pass  # Don't fail on webhook errors
            
            # Interrupt and wait for approval
            resume_value = interrupt([interrupt_payload])
            
            # Parse response
            if isinstance(resume_value, list) and len(resume_value) > 0:
                response: HumanResponse = resume_value[0]
            elif isinstance(resume_value, dict):
                response = resume_value
            else:
                response = {"type": "ignore", "args": None}
            
            response_type = response.get("type", "ignore")
            response_args = response.get("args")
            
            if response_type == "accept":
                return func(*args, **kwargs)
            elif response_type == "edit" and isinstance(response_args, dict):
                edited_args = response_args.get("args", action_args)
                return func(**edited_args)
            elif response_type == "response":
                return {"rejected": True, "human_response": response_args}
            else:  # ignore
                return {"rejected": True, "reason": "Ignored by human"}
        
        # Store metadata on the wrapper for introspection
        wrapper._hitl_action = True
        wrapper._hitl_config = {
            "risk_class": risk_class,
            "anomaly_validators": validators,
        }
        
        return wrapper
    return decorator
