"""
Advanced LangGraph with Multiple Capabilities:
- Tools (Web Search, Calculator, Stock Prices)
- RAG (Document Q&A with FAISS)
- HITL (Human-in-the-Loop for critical actions)
"""

import os
import sqlite3
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

# Configuration
STOCK_API_KEY = os.getenv('STOCK_API_KEY')
VECTOR_STORE_PATH = "vector_store"

# LLM Setup with Ollama
llm = ChatOllama(model='llama3.1:8b', temperature=0.7)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
    Returns comprehensive document excerpts with metadata.
    
    Args:
        query: The question to search for in documents
    """
    if not vector_store:
        return "No documents have been uploaded yet. Please upload documents first."
    
    try:
        # Get more results for better context
        docs = vector_store.similarity_search(query, k=5)
        if not docs:
            return "No relevant information found in documents."
        
        # Build detailed context with metadata
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract metadata
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            
            # Clean and format content
            content = doc.page_content.strip()
            
            context_parts.append(
                f"--- Document Excerpt {i} ---\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{content}\n"
            )
        
        full_context = "\n".join(context_parts)
        
        return (
            f"Found {len(docs)} relevant sections in the document:\n\n"
            f"{full_context}\n\n"
            f"IMPORTANT: Use the information above to answer the user's question. "
            f"The author names, titles, and other details are explicitly mentioned in these excerpts. "
            f"Read carefully and extract the exact information requested."
        )
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

SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools. You can have natural conversations and use tools when appropriate.

**Available Tools:**
- web_search: Search the web for current information
- calculator: Perform mathematical calculations  
- get_stock_price: Get real-time stock prices
- search_documents: Search through uploaded documents
- purchase_stock: Purchase stocks (requires human approval)

**When to Use Tools:**
- Use tools ONLY when the user's question clearly requires them
- For general conversation, greetings, or simple questions, respond naturally WITHOUT using tools
- For document questions: use search_documents
- For math problems: use calculator
- For current events or facts: use web_search
- For stock prices: use get_stock_price

**Examples:**

User: "Hello" or "How are you?" → Respond naturally, NO TOOLS
User: "What's 25 * 47?" → Use calculator tool
User: "Who wrote this paper?" → Use search_documents tool (if documents uploaded)
User: "What's the price of Apple stock?" → Use get_stock_price tool
User: "Tell me a joke" → Respond naturally, NO TOOLS
User: "What happened in the news today?" → Use web_search tool

**Important:**
- Read tool outputs carefully and extract the relevant information
- For stock purchases, wait for human approval before confirming
- Be conversational and friendly
- Don't force tool usage for simple conversations"""


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

conn = sqlite3.connect('chatbot.db', check_same_thread=False)

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
chat_bot = graph.compile(checkpointer=checkpointer)


# ==================== DOCUMENT MANAGEMENT ====================

def add_document(file_path: str) -> str:
    """Add a PDF document to the vector store with enhanced metadata."""
    global vector_store
    
    try:
        # Load PDF with page numbers
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add source filename to metadata
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata['source'] = filename
            # Ensure page number is set
            if 'page' not in doc.metadata:
                doc.metadata['page'] = 'Unknown'
        
        # Split with smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for more precise retrieval
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Preserve metadata in splits
        for split in splits:
            if 'source' not in split.metadata:
                split.metadata['source'] = filename
        
        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)
        
        vector_store.save_local(VECTOR_STORE_PATH)
        
        # Track uploaded documents
        tracking_file = os.path.join(VECTOR_STORE_PATH, "documents.json")
        docs_list = []
        if os.path.exists(tracking_file):
            import json
            with open(tracking_file, 'r') as f:
                docs_list = json.load(f)
        
        # Add new document if not already tracked
        from datetime import datetime
        if not any(d['filename'] == filename for d in docs_list):
            docs_list.append({
                'filename': filename,
                'pages': len(documents),
                'chunks': len(splits),
                'uploaded_at': datetime.now().isoformat()
            })
            
            import json
            with open(tracking_file, 'w') as f:
                json.dump(docs_list, f, indent=2)
        
        return f"Successfully added document: {filename} ({len(splits)} chunks from {len(documents)} pages)"
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