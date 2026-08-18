"""
Advanced LangGraph with Multiple Capabilities:
- Tools (Web Search, Calculator, Stock Prices)
- RAG (Document Q&A with FAISS)
- HITL (Human-in-the-Loop for critical actions)
- MCP (Model Context Protocol - optional)
"""

import os
import sqlite3
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

# Configuration
STOCK_API_KEY = os.getenv('STOCK_API_KEY')
VECTOR_STORE_PATH = "vector_store"

# LLM Setup
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', streaming=True, temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize or load vector store
try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
except:
    vector_store = None


# ==================== TOOLS ====================

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information using DuckDuckGo.
    Use this when you need recent information or facts not in your knowledge base.
    
    Args:
        query: The search query
    """
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=3)
        
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}\n{result['body']}\nSource: {result['href']}\n")
        
        return "\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports basic arithmetic operations: +, -, *, /, **, (), etc.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price for a given ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
    """
    if not STOCK_API_KEY:
        return "Stock API key not configured."
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_pct = quote.get("10. change percent", "N/A")
            
            return f"Stock: {symbol}\nPrice: ${price}\nChange: {change} ({change_pct})"
        else:
            return f"No data found for symbol: {symbol}"
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"


@tool
def search_documents(query: str) -> str:
    """
    Search through uploaded documents using RAG (Retrieval-Augmented Generation).
    Use this to answer questions about specific documents that were uploaded.
    
    Args:
        query: The question to search for in documents
    """
    if not vector_store:
        return "No documents have been uploaded yet. Please upload documents first."
    
    try:
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in documents."
        
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        return f"Found relevant information:\n\n{context}"
    except Exception as e:
        return f"Document search error: {str(e)}"


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing stocks. Requires human approval (HITL).
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        quantity: Number of shares to purchase
    """
    if quantity <= 0:
        return {
            'status': 'error',
            'message': f'Invalid quantity: {quantity}. Must be positive.'
        }
    
    # HITL: Request human approval
    decision = interrupt({
        'type': 'stock_purchase_approval',
        'symbol': symbol,
        'quantity': quantity,
        'action': 'purchase',
        'message': f'Approve purchase of {quantity} shares of {symbol}?'
    })
    
    if isinstance(decision, dict) and decision.get('approved', '').lower() == 'yes':
        return {
            'status': 'success',
            'message': f'✅ Successfully purchased {quantity} shares of {symbol}',
            'symbol': symbol,
            'quantity': quantity,
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'status': 'cancelled',
            'message': f'❌ Purchase of {quantity} shares of {symbol} was declined',
            'symbol': symbol,
            'quantity': quantity
        }


# Bind tools to LLM
tools = [web_search, calculator, get_stock_price, search_documents, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


# ==================== STATE ====================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==================== NODES ====================

SYSTEM_PROMPT = """You are an advanced AI assistant with multiple capabilities:

🔧 **Tools Available:**
- web_search: Search the web for current information
- calculator: Perform mathematical calculations
- get_stock_price: Get real-time stock prices
- search_documents: Query uploaded documents (RAG)
- purchase_stock: Simulate stock purchases (requires approval)

🎯 **Guidelines:**
1. Use tools when appropriate - don't make up information
2. For math questions, use the calculator tool
3. For current events or recent information, use web_search
4. For stock prices, use get_stock_price
5. For questions about uploaded documents, use search_documents
6. Be conversational and helpful
7. If a tool returns an error, explain it clearly to the user

🔒 **HITL (Human-in-the-Loop):**
- Stock purchases require human approval
- Always wait for approval before confirming sensitive actions

Remember: You're here to help, inform, and assist with various tasks!"""


def chat_node(state: ChatState):
    """Main LLM node with system prompt and tool calling capability."""
    messages = state['messages']
    
    # Inject system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    result = llm_with_tools.invoke(messages)
    return {'messages': [result]}


tool_node = ToolNode(tools)


# ==================== DATABASE & CHECKPOINTER ====================

conn = sqlite3.connect('advanced_chatbot.db', check_same_thread=False)

# Create custom table for dynamic titles
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

checkpointer = SqliteSaver(conn=conn)


# ==================== GRAPH ====================

graph = StateGraph(ChatState)

# Add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile
advanced_chat_bot = graph.compile(checkpointer=checkpointer)


# ==================== DOCUMENT MANAGEMENT ====================

def add_document(file_path: str) -> str:
    """Add a PDF document to the vector store."""
    global vector_store
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)
        
        vector_store.save_local(VECTOR_STORE_PATH)
        
        return f"Successfully added document: {os.path.basename(file_path)} ({len(splits)} chunks)"
    except Exception as e:
        return f"Error adding document: {str(e)}"


def clear_documents() -> str:
    """Clear all documents from the vector store."""
    global vector_store
    vector_store = None
    
    try:
        import shutil
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        return "All documents cleared successfully."
    except Exception as e:
        return f"Error clearing documents: {str(e)}"
