
"""
Minimal LangGraph agent to confirm what LangSmith actually logs.

Two nodes: a memory lookup, and a tool call. Run this once, then run
inspect_trace.py to see the raw trace it produced -- no "reasoning"
or "because" field anywhere, just inputs/outputs/metadata per step.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Custom project name using code
os.environ['LANGCHAIN_PROJECT'] = 'trace-demo'

# --- Fake "memory store" -----------------------------------------------
# In your real project this would be your seeded ground-truth scenarios.
MEMORY_STORE = {
    "m1": "user prefers email over phone",
    "m2": "user requested no marketing emails",
}


class AgentState(TypedDict):
    query: str
    retrieved_memory: str
    tool_result: str
    final_answer: str


def memory_lookup(state: AgentState) -> AgentState:
    """Node 1: pretend to search memory for something relevant to the query."""
    # Deliberately simple/deterministic so you can predict the trace.
    if "marketing" in state["query"].lower():
        state["retrieved_memory"] = f"m2: {MEMORY_STORE['m2']}"
    else:
        state["retrieved_memory"] = f"m1: {MEMORY_STORE['m1']}"
    return state


def send_email_tool(state: AgentState) -> AgentState:
    """Node 2: a fake tool call whose output depends on what memory said."""
    if "no marketing emails" in state["retrieved_memory"]:
        state["tool_result"] = "refused: opted out"
    else:
        state["tool_result"] = "sent"
    return state


def generate_answer(state: AgentState) -> AgentState:
    """Node 3: the only node that calls an LLM -- everything above is plain code."""
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    prompt = (
        f"Query: {state['query']}\n"
        f"Retrieved memory: {state['retrieved_memory']}\n"
        f"Tool result: {state['tool_result']}\n"
        "Write one short sentence telling the user what happened."
    )
    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("memory_lookup", memory_lookup)
    graph.add_node("send_email_tool", send_email_tool)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("memory_lookup")
    graph.add_edge("memory_lookup", "send_email_tool")
    graph.add_edge("send_email_tool", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


if __name__ == "__main__":

    app = build_graph()
    result = app.invoke({"query": "send user a promo email about marketing"})

    print("\n--- Agent finished ---")
    print("Retrieved memory:", result["retrieved_memory"])
    print("Tool result:      ", result["tool_result"])
    print("Final answer:     ", result["final_answer"])
    print("\nNow check LangSmith (or run inspect_trace.py) to see the raw trace.")