#!/usr/bin/env python3
"""
Example: Using hitloop with LangGraph's native interrupt()

This example demonstrates the recommended way to use hitloop in production:
- Uses LangGraph's native interrupt() for human-in-the-loop
- Uses LangGraph's checkpointer for persistence
- Compatible with agent-inbox and LangGraph Studio

Run:
    python examples/interrupt_example.py
"""

from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from hitloop import Action, RiskClass, RiskBasedPolicy, TelemetryLogger
from hitloop.langgraph.interrupt_nodes import (
    create_interrupt_gate_node,
    create_interrupt_tool_node,
    should_execute,
)


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict, total=False):
    """State for our example agent."""
    run_id: str
    user_query: str
    proposed_action: Action | dict | None
    approval_decision: Any
    hitl_status: str
    tool_result: dict | None
    final_response: str


# =============================================================================
# Tools
# =============================================================================

def send_email(recipient: str, subject: str, body: str) -> dict:
    """Simulated email sending tool."""
    print(f"📧 Sending email to {recipient}: {subject}")
    return {
        "success": True,
        "message": f"Email sent to {recipient}",
        "email_id": "email-123",
    }


def search_web(query: str) -> dict:
    """Simulated web search tool."""
    print(f"🔍 Searching: {query}")
    return {
        "success": True,
        "results": [f"Result for: {query}"],
    }


def delete_file(filepath: str) -> dict:
    """Simulated file deletion tool."""
    print(f"🗑️ Deleting: {filepath}")
    return {
        "success": True,
        "message": f"File {filepath} deleted",
    }


TOOLS = {
    "send_email": send_email,
    "search_web": search_web,
    "delete_file": delete_file,
}


# =============================================================================
# Nodes
# =============================================================================

def parse_query_node(state: AgentState) -> dict:
    """Simulate LLM parsing user query into an action."""
    query = state.get("user_query", "").lower()
    
    if "email" in query or "send" in query:
        action = Action(
            tool_name="send_email",
            tool_args={
                "recipient": "boss@company.com",
                "subject": "Important Update",
                "body": "Here is the update you requested.",
            },
            risk_class=RiskClass.HIGH,
            rationale="User wants to send an email",
        )
    elif "delete" in query:
        action = Action(
            tool_name="delete_file",
            tool_args={"filepath": "/tmp/important.txt"},
            risk_class=RiskClass.HIGH,
            rationale="User wants to delete a file",
        )
    else:
        action = Action(
            tool_name="search_web",
            tool_args={"query": query},
            risk_class=RiskClass.LOW,
            rationale="User wants to search",
        )
    
    print(f"\n🤖 Proposed action: {action.tool_name} (risk: {action.risk_class.value})")
    return {"proposed_action": action}


def response_node(state: AgentState) -> dict:
    """Generate final response based on tool result."""
    result = state.get("tool_result", {})
    status = state.get("hitl_status", "unknown")
    
    if result and result.get("success"):
        response = f"✅ Done! {result.get('message', result.get('result', 'Success'))}"
    elif status == "auto_approved":
        response = "✅ Action auto-approved and executed"
    else:
        decision = state.get("approval_decision")
        if decision:
            reason = decision.reason if hasattr(decision, "reason") else decision.get("reason", "")
            response = f"❌ Action was rejected: {reason}"
        else:
            response = "❌ Action could not be completed"
    
    return {"final_response": response}


# =============================================================================
# Build Graph
# =============================================================================

def build_graph():
    """Build the LangGraph with interrupt-based HITL."""
    
    # Setup policy and logger
    policy = RiskBasedPolicy(
        high_risk_tools=["send_email", "delete_file"],
        require_approval_for_high=True,
        require_approval_for_medium=False,
    )
    logger = TelemetryLogger(":memory:")
    
    # Create interrupt-based nodes
    hitl_gate = create_interrupt_gate_node(policy, logger)
    tool_executor = create_interrupt_tool_node(TOOLS, logger)
    
    # Build graph
    builder = StateGraph(AgentState)
    
    builder.add_node("parse_query", parse_query_node)
    builder.add_node("hitl_gate", hitl_gate)
    builder.add_node("execute_tool", tool_executor)
    builder.add_node("response", response_node)
    
    # Edges
    builder.add_edge(START, "parse_query")
    builder.add_edge("parse_query", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate",
        should_execute,
        {"execute": "execute_tool", "skip": "response"},
    )
    builder.add_edge("execute_tool", "response")
    builder.add_edge("response", END)
    
    # Compile with checkpointer (required for interrupt)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("hitloop + LangGraph interrupt() Example")
    print("=" * 60)
    
    graph = build_graph()
    
    # Test 1: Low-risk action (auto-approved)
    print("\n--- Test 1: Low-risk action (auto-approved) ---")
    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"run_id": "run-1", "user_query": "search for Python tutorials"},
        config,
    )
    print(f"Result: {result.get('final_response')}")
    print(f"Status: {result.get('hitl_status')}")
    
    # Test 2: High-risk action (requires approval)
    print("\n--- Test 2: High-risk action (requires interrupt) ---")
    config = {"configurable": {"thread_id": "test-2"}}
    result = graph.invoke(
        {"run_id": "run-2", "user_query": "send an email to my boss"},
        config,
    )
    
    # Check if we hit an interrupt
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"]
        print(f"🛑 INTERRUPT: Waiting for human approval")
        print(f"   Payload: {interrupt_info[0].value if interrupt_info else 'N/A'}")
        
        # Simulate human approval
        print("\n   [Simulating human approval...]")
        resumed = graph.invoke(
            Command(resume={"approved": True, "reason": "Looks good", "decided_by": "human:alice"}),
            config,
        )
        print(f"Result after approval: {resumed.get('final_response')}")
        print(f"Status: {resumed.get('hitl_status')}")
    else:
        print(f"Result: {result.get('final_response')}")
    
    # Test 3: High-risk action rejected
    print("\n--- Test 3: High-risk action (rejected) ---")
    config = {"configurable": {"thread_id": "test-3"}}
    result = graph.invoke(
        {"run_id": "run-3", "user_query": "delete the important file"},
        config,
    )
    
    if "__interrupt__" in result:
        print(f"🛑 INTERRUPT: Waiting for human approval")
        
        # Simulate human rejection
        print("   [Simulating human rejection...]")
        resumed = graph.invoke(
            Command(resume={"approved": False, "reason": "Too risky", "decided_by": "human:bob"}),
            config,
        )
        print(f"Result after rejection: {resumed.get('final_response')}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
