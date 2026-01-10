#!/usr/bin/env python3
"""
Example: LangGraph Agent with hitloop Human-in-the-Loop

This example shows how to add human approval to a LangGraph agent.
The agent can perform actions (send emails, delete files) but high-risk
actions require human approval before execution.

Usage:
    # With real LLM (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
    python examples/langgraph_agent.py
    
    # Simulated mode (no API key needed)
    python examples/langgraph_agent.py --simulate
    
    # Auto-approve mode (no prompts)
    python examples/langgraph_agent.py --simulate --auto
"""

import argparse
import asyncio
import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from hitloop import Action, Decision, RiskClass, TelemetryLogger
from hitloop.backends import AutoApproveBackend, CLIBackend
from hitloop.policies import RiskBasedPolicy


# =============================================================================
# 1. Define the State
# =============================================================================

class AgentState(TypedDict):
    """State for our agent with HITL support."""
    messages: Annotated[list, operator.add]
    pending_action: Action | None
    approval_decision: Decision | None
    tool_call_id: str | None


# =============================================================================
# 2. Define Tools (these are the actions the agent can take)
# =============================================================================

TOOLS = {
    "send_email": {
        "description": "Send an email to a recipient",
        "parameters": ["recipient", "subject", "body"],
        "risk": RiskClass.HIGH,  # Requires approval
    },
    "search_web": {
        "description": "Search the web for information",
        "parameters": ["query"],
        "risk": RiskClass.LOW,  # Auto-approved
    },
    "delete_file": {
        "description": "Delete a file from the system",
        "parameters": ["filepath"],
        "risk": RiskClass.HIGH,  # Always requires approval
    },
    "calculate": {
        "description": "Perform a calculation",
        "parameters": ["expression"],
        "risk": RiskClass.LOW,  # Auto-approved
    },
}


def execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "send_email":
        return f"Email sent to {args['recipient']} with subject '{args['subject']}'"
    elif tool_name == "search_web":
        return f"Search results for '{args['query']}': [Result 1, Result 2, Result 3]"
    elif tool_name == "delete_file":
        return f"File '{args['filepath']}' deleted successfully"
    elif tool_name == "calculate":
        try:
            result = eval(args["expression"])  # Simple calc, don't do this in prod!
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {tool_name}"


# =============================================================================
# 3. Simulated LLM (for testing without API keys)
# =============================================================================

class SimulatedLLM:
    """Simulates an LLM that proposes tool calls based on user input."""
    
    def __init__(self):
        self.call_count = 0
    
    def invoke(self, messages: list) -> AIMessage:
        """Simulate LLM response based on the last user message."""
        self.call_count += 1
        
        # Get the last human message
        user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content.lower()
                break
        
        # Check if we already got a tool result
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Done! {messages[-1].content}")
        
        # Simulate tool selection based on keywords
        if "email" in user_msg or "send" in user_msg:
            return AIMessage(
                content="I'll send that email for you.",
                tool_calls=[{
                    "id": f"call_{self.call_count}",
                    "name": "send_email",
                    "args": {
                        "recipient": "bob@example.com",
                        "subject": "Hello from the agent",
                        "body": "This is a test email sent by the AI agent."
                    }
                }]
            )
        elif "search" in user_msg or "find" in user_msg:
            return AIMessage(
                content="I'll search for that.",
                tool_calls=[{
                    "id": f"call_{self.call_count}",
                    "name": "search_web",
                    "args": {"query": user_msg}
                }]
            )
        elif "delete" in user_msg:
            return AIMessage(
                content="I'll delete that file.",
                tool_calls=[{
                    "id": f"call_{self.call_count}",
                    "name": "delete_file",
                    "args": {"filepath": "/tmp/test.txt"}
                }]
            )
        elif "calculate" in user_msg or any(c in user_msg for c in "+-*/"):
            expr = "2 + 2"  # Default
            for word in user_msg.split():
                if any(c.isdigit() for c in word):
                    expr = word
                    break
            return AIMessage(
                content="I'll calculate that.",
                tool_calls=[{
                    "id": f"call_{self.call_count}",
                    "name": "calculate",
                    "args": {"expression": expr}
                }]
            )
        else:
            return AIMessage(content="I can help you send emails, search the web, delete files, or calculate. What would you like to do?")


# =============================================================================
# 4. Build the Graph with HITL
# =============================================================================

def build_agent(
    policy: RiskBasedPolicy,
    backend: CLIBackend | AutoApproveBackend,
    logger: TelemetryLogger,
    use_real_llm: bool = False,
):
    """Build a LangGraph agent with hitloop HITL integration."""
    
    # Initialize LLM
    if use_real_llm:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gpt-4o-mini", temperature=0)
        llm = llm.bind_tools([
            {"name": name, "description": info["description"], "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in info["parameters"]}}}
            for name, info in TOOLS.items()
        ])
    else:
        llm = SimulatedLLM()
    
    # -------------------------------------------------------------------------
    # Node 1: LLM Call - Agent decides what to do
    # -------------------------------------------------------------------------
    def llm_node(state: AgentState) -> dict:
        """LLM decides whether to call a tool."""
        messages = [
            SystemMessage(content="You are a helpful assistant. You can send emails, search the web, delete files, and calculate.")
        ] + state["messages"]
        
        response = llm.invoke(messages)
        
        # If LLM wants to call a tool, create a pending action
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Create hitloop Action with risk classification
            action = Action(
                tool_name=tool_name,
                tool_args=tool_args,
                risk_class=TOOLS.get(tool_name, {}).get("risk", RiskClass.MEDIUM),
                rationale=response.content,
            )
            # Store tool_call_id in state for later use
            tool_call_id = tool_call["id"]
            
            return {
                "messages": [response],
                "pending_action": action,
                "tool_call_id": tool_call_id,
            }
        
        return {
            "messages": [response],
            "pending_action": None,
        }
    
    # -------------------------------------------------------------------------
    # Node 2: HITL Gate - Check if human approval is needed
    # -------------------------------------------------------------------------
    async def hitl_gate_node(state: AgentState) -> dict:
        """Check policy and request approval if needed."""
        action = state.get("pending_action")
        
        if action is None:
            return {"approval_decision": None}
        
        # Check policy
        needs_approval, reason = policy.should_request_approval(action, state)
        
        # Log the proposed action
        logger.log_action_proposed(
            run_id="agent-run",
            action=action,
        )
        
        if needs_approval:
            print(f"\n🛑 HITL Gate: Approval needed for '{action.tool_name}'")
            print(f"   Reason: {reason}")
            print(f"   Risk: {action.risk_class.value}")
            print(f"   Args: {action.tool_args}")
            
            # Request approval from backend
            from hitloop import ApprovalRequest
            request = ApprovalRequest(
                run_id="agent-run",
                action=action,
                policy_name="risk_based",
                policy_reason=reason,
            )
            decision = await backend.request_approval(request)
            
            logger.log_approval_decided(
                run_id="agent-run",
                decision=decision,
            )
            
            return {"approval_decision": decision}
        else:
            # Auto-approve low-risk actions
            decision = Decision(
                action_id=action.id,
                approved=True,
                reason="Auto-approved: low risk action",
                decided_by="policy:auto"
            )
            print(f"\n✅ HITL Gate: Auto-approved '{action.tool_name}' (low risk)")
            return {"approval_decision": decision}
    
    # -------------------------------------------------------------------------
    # Node 3: Tool Execution - Execute if approved
    # -------------------------------------------------------------------------
    def tool_node(state: AgentState) -> dict:
        """Execute the tool if approved."""
        action = state.get("pending_action")
        decision = state.get("approval_decision")
        
        if action is None or decision is None:
            return {"messages": []}
        
        if not decision.approved:
            print(f"\n❌ Tool execution blocked: {decision.reason}")
            tool_call_id = state.get("tool_call_id", "unknown")
            return {
                "messages": [ToolMessage(
                    content=f"Action rejected: {decision.reason}",
                    tool_call_id=tool_call_id
                )],
                "pending_action": None,
                "approval_decision": None,
            }
        
        # Execute the tool
        result = execute_tool(action.tool_name, action.tool_args)
        print(f"\n🔧 Tool executed: {action.tool_name} -> {result}")
        
        tool_call_id = state.get("tool_call_id", "unknown")
        return {
            "messages": [ToolMessage(
                content=result,
                tool_call_id=tool_call_id
            )],
            "pending_action": None,
            "approval_decision": None,
        }
    
    # -------------------------------------------------------------------------
    # Routing Logic
    # -------------------------------------------------------------------------
    def should_continue(state: AgentState) -> Literal["hitl_gate", END]:
        """Route to HITL gate if there's a pending action."""
        if state.get("pending_action"):
            return "hitl_gate"
        return END
    
    def after_hitl(state: AgentState) -> Literal["tool_node", "llm_node", END]:
        """Route after HITL decision."""
        decision = state.get("approval_decision")
        if decision is None:
            return END
        if decision.approved:
            return "tool_node"
        return "llm_node"  # Let LLM know action was rejected
    
    # -------------------------------------------------------------------------
    # Build the Graph
    # -------------------------------------------------------------------------
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("llm_node", llm_node)
    builder.add_node("hitl_gate", hitl_gate_node)
    builder.add_node("tool_node", tool_node)
    
    # Add edges
    builder.add_edge(START, "llm_node")
    builder.add_conditional_edges("llm_node", should_continue, ["hitl_gate", END])
    builder.add_conditional_edges("hitl_gate", after_hitl, ["tool_node", "llm_node", END])
    builder.add_edge("tool_node", "llm_node")
    
    return builder.compile()


# =============================================================================
# 5. Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="LangGraph Agent with hitloop HITL")
    parser.add_argument("--simulate", action="store_true", help="Use simulated LLM (no API key needed)")
    parser.add_argument("--auto", action="store_true", help="Auto-approve all actions")
    parser.add_argument("--query", type=str, default="Send an email to Bob", help="Query for the agent")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LangGraph Agent with hitloop Human-in-the-Loop")
    print("=" * 60)
    
    # Setup hitloop components
    policy = RiskBasedPolicy(
        require_approval_for_high=True,
        high_risk_tools=["send_email", "delete_file"],
    )
    
    backend = AutoApproveBackend() if args.auto else CLIBackend()
    logger = TelemetryLogger("langgraph_agent_traces.db")
    
    # Build agent
    agent = build_agent(
        policy=policy,
        backend=backend,
        logger=logger,
        use_real_llm=not args.simulate,
    )
    
    print(f"\nMode: {'Simulated LLM' if args.simulate else 'Real LLM'}")
    print(f"Approval: {'Auto-approve' if args.auto else 'CLI prompts'}")
    print(f"Query: {args.query}")
    print("-" * 60)
    
    # Run agent
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=args.query)],
        "pending_action": None,
        "approval_decision": None,
        "tool_call_id": None,
    })
    
    print("\n" + "=" * 60)
    print("Final Response:")
    print("=" * 60)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            print(f"🤖 {msg.content}")
    
    print(f"\nTraces saved to: langgraph_agent_traces.db")


if __name__ == "__main__":
    asyncio.run(main())
