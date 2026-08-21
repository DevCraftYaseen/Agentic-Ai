"""
Advanced LangGraph with Multiple Capabilities:
- Tools (Web Search, Calculator, Stock Prices)
- RAG (Document Q&A with FAISS)
- HITL (Human-in-the-Loop for critical actions)
"""

import os
import sqlite3
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
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

# ==================== LLM SETUP ====================
# Two separate model instances at different temperatures:
# - llm_grounded: low temperature, used for tool-calling decisions AND for
#   synthesizing answers from tool/RAG results. Low temp = less hallucination,
#   more faithful to retrieved content.
# - llm_conversational: higher temperature, used ONLY for genuinely tool-free
#   turns (greetings, chit-chat, opinions) where a warmer, more natural voice
#   is desirable and there's no retrieved context to stay faithful to.
llm_grounded = ChatOllama(model='llama3.1:8b', temperature=0.2)
llm_conversational = ChatOllama(model='llama3.1:8b', temperature=0.7)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize or load vector store
try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception:
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
        # Pull more candidates than we'll use so we can filter by relevance
        docs_with_scores = vector_store.similarity_search_with_score(query, k=8)
        if not docs_with_scores:
            return "No relevant information found in documents."

        # FAISS returns L2 distance by default (LOWER score = more similar).
        # This threshold is a starting point — tune it empirically against
        # your own documents/embedding model if you see too many/few results
        # passing the filter (log the scores for a while and eyeball it).
        SCORE_THRESHOLD = 1.2

        relevant = [(doc, score) for doc, score in docs_with_scores if score <= SCORE_THRESHOLD]
        low_confidence = False

        if not relevant:
            # Nothing cleared the bar — fall back to the top 3 raw results,
            # but flag low confidence so the LLM doesn't present them as certain
            relevant = docs_with_scores[:3]
            low_confidence = True
        else:
            relevant = relevant[:5]

        context_parts = []
        for i, (doc, score) in enumerate(relevant, 1):
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()

            context_parts.append(
                f"--- Document Excerpt {i} (relevance score: {score:.3f}, lower=better) ---\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{content}\n"
            )

        full_context = "\n".join(context_parts)

        confidence_note = ""
        if low_confidence:
            confidence_note = (
                "\n\nNOTE: These matches have weak relevance scores — the documents "
                "may not actually contain a good answer to this question. If the "
                "excerpts don't clearly answer it, say so explicitly instead of guessing.\n"
            )

        return (
            f"Found {len(relevant)} relevant section(s) in the document:\n\n"
            f"{full_context}"
            f"{confidence_note}\n"
            f"IMPORTANT: Quote facts (names, titles, numbers, dates) directly from "
            f"the excerpts above rather than paraphrasing from memory. If the excerpts "
            f"don't contain the answer, say so explicitly instead of guessing."
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


# Bind tools ONLY to the grounded (low-temp) model — this is the model that
# makes tool-calling decisions and synthesizes answers from tool/RAG results.
tools = [web_search, calculator, get_stock_price, search_documents, purchase_stock]
llm_with_tools = llm_grounded.bind_tools(tools)


# ==================== STATE ====================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==================== NODES ====================

SYSTEM_PROMPT = """You are a helpful AI assistant. You have tools available, but you should use them RARELY — most requests do not need one.

Before ever calling a tool, ask yourself: "Can I answer this directly from what I already know?" If yes, just answer directly. Only reach for a tool when direct knowledge genuinely cannot cover it.

**Available Tools (use ONLY when clearly necessary):**
- web_search: ONLY for information that changes over time and you could not know from training — breaking news, today's events, very recent releases/prices. NEVER use it for writing tasks, explanations, or general knowledge — you already know how to write an article, story, summary, or explanation about AI, history, science, etc. without searching for it.
- calculator: ONLY for arithmetic you are not fully confident doing mentally.
- get_stock_price: ONLY when the user explicitly asks for a current stock price.
- search_documents: ONLY when the user's question is about content from a document they uploaded.
- purchase_stock: ONLY when the user explicitly asks to buy/purchase shares.

**Do NOT use any tool for:**
- Greetings, small talk, introductions ("hi", "who are you", "how's it going")
- Writing requests: articles, essays, stories, poems, summaries, emails, code — write these yourself directly
- Explaining concepts, definitions, how-to questions, general knowledge
- Opinions, brainstorming, casual advice
- Math you can do without a calculator

**Examples:**
User: "Hi I am Yaseen" → respond directly, NO TOOLS
User: "Write me a 200 word article on AI" → write it yourself directly, NO TOOLS — you already know about AI, no search needed
User: "Explain how neural networks work" → respond directly, NO TOOLS
User: "Tell me a joke" → respond directly, NO TOOLS
User: "What's 25 * 47?" → calculator
User: "Who wrote this paper?" (with a document uploaded) → search_documents
User: "What's the price of Apple stock right now?" → get_stock_price
User: "What happened in the news today?" → web_search
User: "Buy me 10 shares of AAPL" → purchase_stock

**When genuinely unsure whether a tool is needed, default to NOT using one and answer directly.**
- Read tool outputs carefully and extract the relevant information when you do use one
- For stock purchases, wait for human approval before confirming"""


async def _stream_and_merge(model, messages, config: RunnableConfig):
    """
    Stream a model's response chunk by chunk and return the fully merged
    AIMessageChunk.

    This MUST be async (model.astream + async for), not sync model.stream().
    LangGraph's stream_mode="messages" only forwards tokens live when nodes
    run asynchronously — with a synchronous node, LangGraph appears to
    collect everything produced during that node's execution and only
    release it once the node function returns, which looks identical to
    "no streaming" from the client's perspective even though the model
    itself streamed correctly under the hood.
    """
    full = None
    async for chunk in model.astream(messages, config=config):
        full = chunk if full is None else full + chunk
    return full


async def chat_node(state: ChatState, config: RunnableConfig):
    """
    Main routing node (now async — required for live token forwarding).

    1. If we just got a tool result back, stream the grounded model's
       synthesis directly.
    2. Otherwise, run a single non-streamed ainvoke() purely to decide
       whether a tool call is needed — invisible to the client either way,
       sync or async, since only .stream()/.astream() calls get forwarded.
       If it produced a real tool call, that's the result. If not, stream
       the actual answer from the conversational model.
    """
    messages = state['messages']

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    just_used_tool = len(messages) > 0 and isinstance(messages[-1], ToolMessage)

    if just_used_tool:
        result = await _stream_and_merge(llm_with_tools, messages, config)
        return {'messages': [result]}

    routing_result = await llm_with_tools.ainvoke(messages, config=config)

    if getattr(routing_result, 'tool_calls', None):
        return {'messages': [routing_result]}

    chat_messages = [SystemMessage(content=SYSTEM_PROMPT)] + [
        m for m in messages if not isinstance(m, SystemMessage)
    ]
    result = await _stream_and_merge(llm_conversational, chat_messages, config)
    return {'messages': [result]}


tool_node = ToolNode(tools)


tool_node = ToolNode(tools)


# ==================== DATABASE (thread titles only) ====================
# This connection is ONLY for the plain chat_titles table — unrelated to
# LangGraph's own checkpointing. It stays synchronous since it's just used
# for simple cursor.execute() calls in main.py's title/thread helpers.
# Deliberately a SEPARATE db file from the checkpointer's own database
# below, to avoid two different sqlite drivers (sync + aiosqlite) locking
# the same file concurrently.

conn = sqlite3.connect('chatbot.db', check_same_thread=False)

cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()


# ==================== GRAPH (built but NOT compiled here) ====================
# We only build the graph's shape here. Compilation requires a checkpointer,
# and AsyncSqliteSaver needs a real async DB connection — that can't be
# created at plain module-import time (there's no running event loop yet).
# main.py compiles this into `chat_bot` inside its FastAPI lifespan handler,
# where an async context is available.

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


# ==================== DOCUMENT MANAGEMENT ====================

def add_document(file_path: str) -> str:
    """Add a PDF document to the vector store with enhanced metadata."""
    global vector_store

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata['source'] = filename
            if 'page' not in doc.metadata:
                doc.metadata['page'] = 'Unknown'

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        for split in splits:
            if 'source' not in split.metadata:
                split.metadata['source'] = filename

        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)

        vector_store.save_local(VECTOR_STORE_PATH)

        tracking_file = os.path.join(VECTOR_STORE_PATH, "documents.json")
        docs_list = []
        if os.path.exists(tracking_file):
            import json
            with open(tracking_file, 'r') as f:
                docs_list = json.load(f)

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