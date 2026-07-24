from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import os

load_dotenv()

# API Keys 
STOCK_API_KEY = os.getenv('STOCK_API_KEY')
CURRENCY_CONVERTER_API = os.getenv('CURRENCY_CONVERTER_API')

# Custom project name using code
os.environ['LANGCHAIN_PROJECT'] = 'Chatbot with Tools'

# 1. LLM
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# 2. Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

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

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def fetch_rate(base_curr: str, target_curr: str):
    """This function fetches the currency conversion factor between a given base currency and a target currency."""
    url = f"https://v6.exchangerate-api.com/v6/{CURRENCY_CONVERTER_API}/pair/{base_curr}/{target_curr}"
    response = requests.get(url)
    return response.json()

@tool
def convert(base_curr_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Given the conversion rate, this function calculates the target currency value from a given base_currency_value"""
    return base_curr_value * conversion_rate

tools = [search_tool, get_stock_price, calculator, fetch_rate, convert]
llm_with_tools = llm.bind_tools(tools)

# 3. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# 6. Graph
graph = StateGraph(ChatState)

# Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile Graph
chat_bot = graph.compile(checkpointer=checkpointer)

# 7. Helper
def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)