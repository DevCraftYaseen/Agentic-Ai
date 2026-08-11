from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import asyncio
import threading

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# -------------------
# 2. MCP Client & Tools
# -------------------
SERVERS = {
    "math-server": {
        'transport': 'stdio',
        "command": "fastmcp",
        "args": [
            "run",
            "D:\\MCP\\math-mcp-server\\src\\math_mcp_server\\__init__.py"
        ]
    },
    "expense-tracker": {
        'transport': 'stdio',
        "command": "fastmcp",
        "args": [
            "run",
            "D:\\MCP\\demo-mcp\\src\\demo_mcp\\__init__.py"
        ]
    },
}

client = MultiServerMCPClient(SERVERS)


def load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools synchronously on the backend loop."""
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []


mcp_tools = load_mcp_tools()

tools = [*mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    from langchain_core.messages import SystemMessage
    
    messages = state["messages"]
    
    # Add system message to make it a general-purpose assistant
    system_message = SystemMessage(
        content="""You are a helpful AI assistant that can chat about any topic. 
You have access to specialized tools for:
- Math operations (add, subtract, multiply, divide, modulus)
- Expense tracking (add, list, edit, delete transactions, get balance, summarize expenses)

Use these tools ONLY when the user asks questions related to these specific tasks. 
For all other conversations (general questions, advice, chitchat, etc.), respond naturally without using tools.

Be friendly, helpful, and conversational on any topic the user wants to discuss."""
    )
    
    # Insert system message at the beginning if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_message] + messages
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())


# # -------------------
# # 8. Test Function (Optional)
# # -------------------
# async def main():
#     """Test the chatbot directly."""
    
#     # Test 1: General conversation
#     print("=== Test 1: General Conversation ===")
#     result1 = await chatbot.ainvoke(
#         {
#             'messages': [HumanMessage(content='Tell me about the weather and seasons.')]
#         },
#         config={"configurable": {"thread_id": "test1"}}
#     )
#     print(result1['messages'][-1].content)
#     print("\n")
    
#     # Test 2: Math operation
#     print("=== Test 2: Math Operation ===")
#     result2 = await chatbot.ainvoke(
#         {
#             'messages': [HumanMessage(content='What is 456 multiplied by 789?')]
#         },
#         config={"configurable": {"thread_id": "test2"}}
#     )
#     print(result2['messages'][-1].content)
#     print("\n")
    
#     # Test 3: Expense tracking
#     print("=== Test 3: Expense Tracking ===")
#     result3 = await chatbot.ainvoke(
#         {
#             'messages': [HumanMessage(content='What are my expenses for August 2026?')]
#         },
#         config={"configurable": {"thread_id": "test3"}}
#     )
#     print(result3['messages'][-1].content)


# if __name__ == '__main__':
#     asyncio.run(main())
