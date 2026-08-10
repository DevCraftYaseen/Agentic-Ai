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

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# Checkpointer
# conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Build graph
def build_graph():

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
    chatbot = build_graph()

    result = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content = 'Find the modulus of 132345 and 23 and give answer like a cricker commentator.')] 
        },
        config={"configurable": {"thread_id": "1"}}
    )

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())