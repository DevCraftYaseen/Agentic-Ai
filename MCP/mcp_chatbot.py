from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

SERVERS = {
    "math-server": {
        'transport': 'stdio',
        "command": "fastmcp",
        "args": [
            "run",
            "D:\\MCP\\math-mcp-server\\src\\math_mcp_server\\__init__.py"
        ]
    }
}

client = MultiServerMCPClient(SERVERS)

# Checkpointer
# conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Build graph
async def build_graph():

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # Graph
    graph = StateGraph(ChatState)

    # Nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools', 'chat_node')

    # Compile Graph
    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content = 'Find the modulus of 132345 and 23 and give answer like a cricker commentator.')] 
        },
        config={"configurable": {"thread_id": "1"}}
    )

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())