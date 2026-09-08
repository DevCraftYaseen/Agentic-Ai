"""
LangGraph chatbot with CodeCraft API, memory, and web search capabilities.
Features:
- Long-term memory with SQLite (InMemoryStore for simplicity)
- Web search using Tavily
- Streaming responses
- Thread-based conversations
"""

import operator
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from openai import OpenAI
import os


# ============================================================
# State
# ============================================================
class State(TypedDict):
    messages: Annotated[List, operator.add]
    user_id: str
    relevant_memories: List[str]
    search_results: Optional[str]


# ============================================================
# Memory Store Setup
# ============================================================
def get_memory_store():
    """Initialize in-memory store for memories"""
    return InMemoryStore()


# ============================================================
# Tools
# ============================================================
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using Tavily API"""
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        
        if not results:
            return "No search results found."
        
        formatted = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            formatted.append(f"**{title}**\n{content}\nSource: {url}")
        
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================
# Nodes
# ============================================================
def retrieve_memories(state: State, store) -> dict:
    """Retrieve relevant memories for the user"""
    user_id = state.get("user_id", "default_user")
    namespace = ("memories", user_id)
    
    try:
        # Get recent memories
        memories = store.search(namespace, limit=5)
        relevant = [mem.value.get("content", "") for mem in memories if mem.value.get("content")]
        return {"relevant_memories": relevant}
    except Exception as e:
        print(f"Memory retrieval error: {e}")
        return {"relevant_memories": []}


def decide_action(state: State) -> str:
    """Decide if web search is needed"""
    last_message = state["messages"][-1].content.lower()
    
    # Search triggers
    search_keywords = [
        "search", "look up", "find", "what is", "who is", "when did",
        "latest", "recent", "news", "current", "today", "2026"
    ]
    
    if any(keyword in last_message for keyword in search_keywords):
        return "search"
    return "respond"


def search_web(state: State) -> dict:
    """Perform web search"""
    query = state["messages"][-1].content
    results = web_search(query)
    return {"search_results": results}


def generate_response(state: State, store) -> dict:
    """Generate AI response using CodeCraft API"""
    # Initialize CodeCraft client
    client = OpenAI(
        api_key=os.getenv("CODECRAFT_API_KEY"),
        base_url=os.getenv("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")
    )
    model = os.getenv("CODECRAFT_MODEL", "gpt-5.6-sol")
    
    # Build context
    memories = state.get("relevant_memories", [])
    search_results = state.get("search_results")
    
    system_prompt = """You are a helpful AI assistant with access to the user's conversation history and web search.

Your capabilities:
- Remember user preferences and past conversations
- Search the web for current information when needed
- Provide accurate, helpful, and contextual responses

When using memories, naturally incorporate relevant context.
When using search results, cite sources when appropriate."""
    
    # Add memories context
    if memories:
        memory_context = "\n\n**Relevant memories about this user:**\n" + "\n".join(f"- {m}" for m in memories)
        system_prompt += memory_context
    
    # Add search results
    if search_results:
        system_prompt += f"\n\n**Web Search Results:**\n{search_results}"
    
    # Prepare messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (last 10 messages)
    for msg in state["messages"][-10:]:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    # Generate response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    ai_response = response.choices[0].message.content
    
    # Extract and store memories
    extract_and_store_memories(state, ai_response, store)
    
    return {"messages": [AIMessage(content=ai_response)]}


def extract_and_store_memories(state: State, ai_response: str, store):
    """Extract important information and store as memories"""
    user_id = state.get("user_id", "default_user")
    namespace = ("memories", user_id)
    
    # Get last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    if not last_user_msg:
        return
    
    # Use LLM to extract memories
    client = OpenAI(
        api_key=os.getenv("CODECRAFT_API_KEY"),
        base_url=os.getenv("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")
    )
    
    extraction_prompt = f"""Extract factual information about the user from this conversation that should be remembered for future interactions.

User said: {last_user_msg}
Assistant said: {ai_response}

Extract ONLY clear, factual information like:
- Name: [name]
- Preference: [preference]
- Fact: [fact]
- Interest: [interest]

Return ONE memory per line in format "Type: information"
If nothing important to remember, return "NONE"
"""
    
    try:
        extraction = client.chat.completions.create(
            model=os.getenv("CODECRAFT_MODEL", "gpt-5.6-sol"),
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            max_tokens=200
        )
        
        extracted = extraction.choices[0].message.content.strip()
        
        if extracted != "NONE" and extracted:
            # Store each extracted memory
            for line in extracted.split("\n"):
                line = line.strip()
                if line and ":" in line:
                    memory_id = f"mem_{user_id}_{datetime.now().timestamp()}"
                    store.put(
                        namespace,
                        memory_id,
                        {
                            "content": line,
                            "timestamp": datetime.now().isoformat(),
                            "source": "conversation"
                        }
                    )
    except Exception as e:
        print(f"Memory extraction error: {e}")


def trim_messages(state: State) -> dict:
    """Keep only recent messages to manage context window"""
    messages = state["messages"]
    
    # Keep last 20 messages
    if len(messages) > 20:
        # Keep system messages and recent conversation
        to_remove = messages[:-20]
        return {"messages": [RemoveMessage(id=msg.id) for msg in to_remove if hasattr(msg, 'id')]}
    
    return {}


# ============================================================
# Build Graph
# ============================================================
def build_graph():
    """Build the chatbot graph (synchronous wrapper)"""
    # Initialize store
    store = get_memory_store()
    
    # Create graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("retrieve_memories", lambda state: retrieve_memories(state, store))
    graph.add_node("search_web", search_web)
    graph.add_node("generate_response", lambda state: generate_response(state, store))
    graph.add_node("trim_messages", trim_messages)
    
    # Add edges
    graph.add_edge(START, "retrieve_memories")
    graph.add_conditional_edges(
        "retrieve_memories",
        decide_action,
        {
            "search": "search_web",
            "respond": "generate_response"
        }
    )
    graph.add_edge("search_web", "generate_response")
    graph.add_edge("generate_response", "trim_messages")
    graph.add_edge("trim_messages", END)
    
    # Compile without checkpointer first (add it later in FastAPI)
    return graph.compile(store=store)


# ============================================================
# Export
# ============================================================
app = build_graph()
