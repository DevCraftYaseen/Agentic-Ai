from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio
import aiosqlite
import os
import threading
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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
# 1. LLM Configuration
# -------------------

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
# llm = ChatOllama(model='qwen2.5-coder:7b', temperature=0)

# -------------------
# 2. RAG Components
# -------------------

# Embedding Model - Using Ollama's nomic-embed-text
embedding_model = OllamaEmbeddings(model='nomic-embed-text')

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Smaller chunks = more precise retrieval
    chunk_overlap=200,  # Overlap helps maintain context
    separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting
)

# Storage paths - Use absolute paths to avoid nesting issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

# Global vector store (will be updated when files are added/removed)
vector_store = None
retriever = None


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a readable string for the LLM.
    """
    if not docs:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Include source filename if available
        source = doc.metadata.get('source', 'unknown')
        filename = os.path.basename(source)
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"[Source: {filename} - Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def load_pdf_file(file_path: str) -> List[Document]:
    """
    Load a single PDF file and split it into chunks.
    """
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content loaded from {file_path}")
        
        # Split into chunks
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []


def rebuild_vector_store(uploaded_files: List[str]) -> tuple[bool, str]:
    """
    Rebuild the vector store from all uploaded files.
    
    Args:
        uploaded_files: List of file paths to index
        
    Returns:
        (success: bool, message: str)
    """
    global vector_store, retriever
    
    try:
        if not uploaded_files:
            print("⚠️ No files to index")
            vector_store = None
            retriever = None
            return False, "No files to index"
        
        print(f"🔮 Building vector store from {len(uploaded_files)} file(s)...")
        
        all_chunks = []
        failed_files = []
        
        for file_path in uploaded_files:
            if os.path.exists(file_path):
                chunks = load_pdf_file(file_path)
                if chunks:
                    all_chunks.extend(chunks)
                    print(f"  ✅ Loaded {len(chunks)} chunks from {os.path.basename(file_path)}")
                else:
                    failed_files.append(os.path.basename(file_path))
                    print(f"  ❌ Failed to load {os.path.basename(file_path)}")
        
        if not all_chunks:
            error_msg = "Failed to extract text from PDF"
            if failed_files:
                error_msg += f": {', '.join(failed_files)}"
            print(f"❌ {error_msg}")
            vector_store = None
            retriever = None
            return False, error_msg
        
        print(f"📊 Total chunks: {len(all_chunks)}")
        print("🔮 Creating embeddings...")
        
        # Create vector store
        vector_store = FAISS.from_documents(all_chunks, embedding=embedding_model)
        retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 4}
        )
        
        print("✅ Vector store created successfully")
        return True, "Successfully indexed document"
    
    except Exception as e:
        error_msg = f"Indexing error: {str(e)}"
        print(f"❌ {error_msg}")
        vector_store = None
        retriever = None
        return False, error_msg


def get_uploaded_files() -> List[str]:
    """
    Get list of all uploaded PDF files.
    """
    if not os.path.exists(UPLOAD_DIR):
        return []
    
    files = [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith('.pdf')
    ]
    return sorted(files)


def add_file(file_path: str) -> tuple[bool, str]:
    """
    Add a new file to the RAG system.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Try to build vector store with this file included
        uploaded_files = get_uploaded_files()
        if file_path not in uploaded_files:
            uploaded_files.append(file_path)
        
        success, message = rebuild_vector_store(uploaded_files)
        
        if success:
            return True, "File indexed successfully!"
        else:
            # Rollback: remove the file if indexing failed
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🔄 Rolled back: removed {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ Could not remove file during rollback: {e}")
            return False, f"Failed to index: {message}"
    
    except Exception as e:
        # Rollback on any error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return False, f"Error: {str(e)}"


def remove_file(file_path: str) -> tuple[bool, str]:
    """
    Remove a file from the RAG system.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {os.path.basename(file_path)}"
        
        # Delete the file
        os.remove(file_path)
        print(f"🗑️ Removed file: {os.path.basename(file_path)}")
        
        # Rebuild vector store with remaining files
        uploaded_files = get_uploaded_files()
        
        if uploaded_files:
            success, message = rebuild_vector_store(uploaded_files)
            if success:
                return True, f"Removed '{os.path.basename(file_path)}' successfully!"
            else:
                return True, f"Removed '{os.path.basename(file_path)}' but failed to rebuild index: {message}"
        else:
            # No files left
            global vector_store, retriever
            vector_store = None
            retriever = None
            return True, "All documents removed."
    
    except Exception as e:
        return False, f"Error removing file: {str(e)}"


# -------------------
# 3. Tools
# -------------------

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div, mod
    
    Args:
        first_num: First number
        second_num: Second number
        operation: One of 'add', 'sub', 'mul', 'div', 'mod'
    """
    try:
        operation = operation.lower()
        
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
        elif operation == "mod":
            if second_num == 0:
                return {"error": "Modulus by zero is not allowed"}
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation '{operation}'. Use: add, sub, mul, div, mod"}
        
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
            "formatted": f"{first_num} {operation} {second_num} = {result}"
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded documents for information relevant to the query.
    Use this when the user asks questions about the uploaded documents.
    
    Args:
        query: The search query or question about the documents
        
    Returns:
        Relevant excerpts from the documents
    """
    if retriever is None:
        return "No documents have been uploaded yet. Please upload PDF files first."
    
    try:
        retrieved_docs = retriever.invoke(query)
        formatted_context = format_docs(retrieved_docs)
        return formatted_context
    
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


# Bind tools to LLM
tools = [calculator, search_documents]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="rag_chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
async def chat_node(state: ChatState):
    """
    LLM node that processes messages and decides on tool usage.
    """
    messages = state["messages"]
    
    # Add system message with instructions
    uploaded_files = get_uploaded_files()
    file_list = ", ".join([os.path.basename(f) for f in uploaded_files]) if uploaded_files else "None"
    
    system_message = SystemMessage(
        content=f"""You are a helpful AI assistant that can chat about any topic.

You have access to specialized tools:
- 'search_documents' tool: Searches uploaded PDF documents
- 'calculator' tool: Performs math operations

Currently uploaded documents: {file_list}

IMPORTANT INSTRUCTIONS:
- For questions about the uploaded documents → Use 'search_documents' tool
- For math calculations → Use 'calculator' tool  
- For ANY other topic (general knowledge, advice, conversation) → Respond naturally WITHOUT using tools

Be friendly, helpful, and conversational. You can discuss any topic the user wants!"""
    )
    
    # Insert system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_message] + messages
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# Build graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Define edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper Functions
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())


# -------------------
# 8. Initialize on Import
# -------------------
# Load existing files on startup
uploaded_files = get_uploaded_files()
if uploaded_files:
    print(f"📂 Found {len(uploaded_files)} existing file(s)")
    success, message = rebuild_vector_store(uploaded_files)
    if not success:
        print(f"⚠️ Failed to load existing files: {message}")
else:
    print("📂 No existing files found")
