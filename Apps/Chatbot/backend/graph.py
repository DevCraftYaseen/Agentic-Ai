"""
Advanced LangGraph with Multiple Capabilities:
- Tools (Web Search, Calculator, Stock Prices)
- RAG (Document Q&A with FAISS)
- HITL (Human-in-the-Loop for critical actions)
- STM (Short-Term Memory with message trimming)
- LTM (Long-Term Memory with PostgreSQL persistence)
"""

import os
import sqlite3
import uuid
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages, RemoveMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.store.base import BaseStore
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
FIXED_USER_ID = "default-user"  # Fixed user ID for LTM (will be dynamic later)
MESSAGE_THRESHOLD = 15  # STM: Trigger summarization after 15 messages

# ==================== LLM SETUP ====================
llm_grounded = ChatOllama(model='qwen3.5:9b', temperature=0.2)
llm_conversational = ChatOllama(model='qwen3.5:9b', temperature=0.7)
memory_llm = ChatOllama(model='qwen3.5:9b', temperature=0)  # For LTM extraction
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


tools = [web_search, calculator, get_stock_price, search_documents, purchase_stock]
llm_with_tools = llm_grounded.bind_tools(tools)


# ==================== LTM MEMORY STRUCTURES ====================

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")


class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")


memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """Extract user information worth remembering long-term from the message below.

Current stored memories:
{user_details_content}

Rules:
- Extract ONLY factual information: name, profession, interests, projects, preferences
- Keep each memory as a single, clear sentence
- Set is_new=true ONLY if this is genuinely NEW information not already stored
- If information is similar to existing memory, set is_new=false
- Ignore questions, greetings, or temporary statements
- If nothing worth storing, set should_write=false

Examples:
- "Hi, I'm Yaseen" → "Name: Yaseen" (is_new=true if not stored)
- "I love Python" → "Prefers Python programming language" (is_new=true)
- "How are you?" → Nothing to store (should_write=false)
- "I work on AI" when "Works as AI engineer" exists → is_new=false
"""


# ==================== STATE ====================

class ChatState(MessagesState):
    """Extended state with STM summary and LTM context"""
    summary: str = ""  # STM: Conversation summary


# ==================== NODES ====================

def get_system_prompt(user_context: str = "") -> str:
    """Generate system prompt with optional user context from LTM"""
    base_prompt = """You are a helpful AI assistant. You have tools available, but you should use them RARELY — most requests do not need one.

Before ever calling a tool, ask yourself: "Can I answer this directly from what I already know?" If yes, just answer directly. Only reach for a tool when direct knowledge genuinely cannot cover it."""
    
    if user_context:
        base_prompt += f"\n\n{user_context}"
    
    base_prompt += """

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

**Guidelines:**
- Use the user's name naturally in conversation when you know it
- Reference their interests, projects, or preferences when relevant
- Keep responses concise and conversational
- Be helpful and direct

**When genuinely unsure whether a tool is needed, default to NOT using one and answer directly.**
- Read tool outputs carefully and extract the relevant information when you do use one
- For stock purchases, wait for human approval before confirming"""
    
    return base_prompt


def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    """LTM node: Extract and store user information"""
    user_id = config["configurable"].get("user_id", FIXED_USER_ID)
    ns = ("user", user_id, "details")

    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {}

    last_text = last_message.content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}


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
    """Stream a model's response chunk by chunk and return the fully merged AIMessageChunk."""
    full = None
    async for chunk in model.astream(messages, config=config):
        full = chunk if full is None else full + chunk
    return full


async def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    """
    Main chat node with:
    - STM: Uses summary if available, keeps recent messages
    - LTM: User context injection for personalization
    - Tool routing and streaming
    """
    messages = []
    
    # STM: Add conversation summary if exists
    if state.get('summary'):
        messages.append(SystemMessage(
            content=f'Previous Conversation Summary:\n{state["summary"]}'
        ))
    
    # LTM: Get user context
    user_id = config["configurable"].get("user_id", FIXED_USER_ID)
    ns = ("user", user_id, "details")
    items = store.search(ns)
    
    user_context = ""
    if items:
        user_details = "\n".join(f"- {it.value.get('data', '')}" for it in items)
        user_context = f"**User Context (use naturally in conversation):**\n{user_details}\n"
    
    # Inject system prompt with user context
    system_prompt = get_system_prompt(user_context)
    messages.append(SystemMessage(content=system_prompt))
    
    # Add current conversation messages
    messages.extend(state['messages'])

    just_used_tool = len(messages) > 0 and isinstance(messages[-1], ToolMessage)

    if just_used_tool:
        result = await _stream_and_merge(llm_with_tools, messages, config)
        return {'messages': [result]}

    routing_result = await llm_with_tools.ainvoke(messages, config=config)

    if getattr(routing_result, 'tool_calls', None):
        return {'messages': [routing_result]}

    result = await _stream_and_merge(llm_conversational, messages, config)
    return {'messages': [result]}


async def summarization_node(state: ChatState):
    """STM node: Summarize conversation and remove old messages"""
    existing_summary = state.get('summary', '')

    if existing_summary:
        prompt = (
            f'Existing Summary:\n{existing_summary}\n\n'
            'Extend the summary by incorporating the new conversation above. '
            'Keep it concise and focus on key points.'
        )
    else:
        prompt = (
            'Summarize the conversation above in a concise manner. '
            'Include key topics discussed, decisions made, and important context.'
        )

    summary_messages = state['messages'] + [HumanMessage(content=prompt)]
    response = await llm_conversational.ainvoke(summary_messages)

    # Keep last 7 messages, delete the rest
    messages_to_del = state['messages'][:-7]
    
    return {
        'summary': response.content,
        'messages': [RemoveMessage(id=m.id) for m in messages_to_del]
    }


def should_summarize(state: ChatState) -> bool:
    """Decide whether to summarize based on message count"""
    return len(state['messages']) > MESSAGE_THRESHOLD


def route_after_chat(state: ChatState):
    """Route after chat: check for tools first, then summarization"""
    # Check if tools are needed
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    
    # No tools needed - check if summarization needed
    if len(state['messages']) > MESSAGE_THRESHOLD:
        return 'summarize'
    
    return END


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

graph = StateGraph(ChatState)

# Add nodes
graph.add_node('remember', remember_node)  # LTM memory extraction
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_node('summarize', summarization_node)  # STM summarization

# Add edges
graph.add_edge(START, 'remember')  # First extract memories (LTM)
graph.add_edge('remember', 'chat_node')  # Then chat
graph.add_edge('tools', 'chat_node')  # After tools, back to chat

# Custom routing: check tools first, then summarization, then end
graph.add_conditional_edges(
    'chat_node',
    route_after_chat,
    ['tools', 'summarize', END]
)

graph.add_edge('summarize', END)


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