import queue
import uuid
import os
import shutil

import streamlit as st
from rag_backend import (
    chatbot, 
    retrieve_all_threads, 
    submit_async_task,
    get_uploaded_files,
    add_file,
    remove_file,
    UPLOAD_DIR
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================== Page Configuration ===========================
st.set_page_config(
    page_title="RAG Chatbot with File Upload",
    page_icon="📚",
    layout="wide"
)

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    """Reset chat only if current chat has messages"""
    # Silently prevent creating new empty chat
    if not st.session_state["message_history"]:
        return  # Don't create new chat, don't show message
    
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.rerun()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """Load conversation and filter out tool messages."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    
    # Filter out ToolMessage objects - only show Human and AI messages
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            filtered_messages.append(msg)
    
    return filtered_messages


def save_uploaded_file(uploaded_file):
    """Save uploaded file to the uploads directory."""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Check if file already exists
        if os.path.exists(file_path):
            return None, f"File '{uploaded_file.name}' already exists! Please delete it first or rename your file."
        
        # Validate file
        if uploaded_file.size == 0:
            return None, "File is empty (0 bytes)!"
        
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
            return None, f"File too large ({format_file_size(uploaded_file.size)})! Maximum size is 50MB."
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path, None
    except Exception as e:
        return None, f"Error saving file: {str(e)}"


def format_file_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "uploaded_files_list" not in st.session_state:
    st.session_state["uploaded_files_list"] = get_uploaded_files()

if "indexing_in_progress" not in st.session_state:
    st.session_state["indexing_in_progress"] = False

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
with st.sidebar:
    st.title("📚 RAG Chatbot")
    
    # -------- Chat Management --------
    st.header("💬 Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    
    # Show recent conversations
    if st.session_state["chat_threads"]:
        st.subheader("Recent Chats")
        for thread_id in st.session_state["chat_threads"][-5:][::-1]:  # Show last 5
            # Truncate thread_id for display
            display_id = f"Chat {thread_id[:8]}..."
            if st.button(display_id, key=f"thread_{thread_id}", use_container_width=True):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                
                temp_messages = []
                for msg in messages:
                    # Only include Human and AI messages (tool messages already filtered in load_conversation)
                    if isinstance(msg, HumanMessage):
                        role = "user"
                        content = msg.content
                    elif isinstance(msg, AIMessage):
                        role = "assistant"
                        content = msg.content
                    else:
                        # Skip any other message types (ToolMessage, SystemMessage, etc.)
                        continue
                    
                    # Skip empty messages
                    if content and content.strip():
                        temp_messages.append({"role": role, "content": content})
                
                st.session_state["message_history"] = temp_messages
                st.rerun()
    
    st.divider()
    
    # -------- File Management --------
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF file to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Add to Knowledge Base", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                # Save file first
                file_path, error = save_uploaded_file(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    # File saved successfully, now index it
                    progress_text = st.empty()
                    progress_text.info("📄 Indexing document...")
                    
                    # Add file and rebuild vector store
                    success, message = add_file(file_path)
                    progress_text.empty()
                    
                    if success:
                        st.session_state["uploaded_files_list"] = get_uploaded_files()
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
    
    st.divider()
    
    # -------- Uploaded Files List --------
    st.subheader("📚 Uploaded Documents")
    
    uploaded_files = st.session_state["uploaded_files_list"]
    
    if uploaded_files:
        st.caption(f"Total: {len(uploaded_files)} document(s)")
        
        for file_path in uploaded_files:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text(f"📄 {filename}")
                st.caption(f"Size: {format_file_size(file_size)}")
            
            with col2:
                if st.button("🗑️", key=f"delete_{filename}", help="Delete this file"):
                    with st.spinner(f"Removing '{filename}'..."):
                        success, message = remove_file(file_path)
                        
                        if success:
                            st.session_state["uploaded_files_list"] = get_uploaded_files()
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            st.divider()
    else:
        st.info("No documents uploaded yet. Upload a PDF to get started!")

# ============================ Main Chat Area ============================
st.title("💬 Chat with Your Documents")

if not st.session_state["uploaded_files_list"]:
    st.info("👈 Upload PDF documents using the sidebar to enable document search")
else:
    st.success(f"✅ {len(st.session_state['uploaded_files_list'])} document(s) loaded and ready!")

st.divider()

# Render chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything about your documents or general questions...")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder for status updates
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Show tool usage status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    
                    # Map tool names to friendly names
                    tool_display_names = {
                        "search_documents": "🔍 Searching Documents",
                        "web_search": "🌐 Searching Web",
                        "calculator": "🧮 Calculating"
                    }
                    display_name = tool_display_names.get(tool_name, f"🔧 Using {tool_name}")
                    
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            display_name, expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=display_name,
                            state="running",
                            expanded=True,
                        )

                # Stream assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Complete", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

# ============================ Footer ============================
st.divider()
with st.expander("ℹ️ How to use"):
    st.markdown("""
    ### 📖 Usage Guide
    
    **1. Upload Documents:**
    - Use the sidebar to upload PDF files
    - Click "Add to Knowledge Base" to index the document
    - Wait for indexing to complete
    
    **2. Chat:**
    - Ask questions about your uploaded documents
    - The AI will search through your documents to find relevant information
    - You can also ask general questions or perform calculations
    
    **3. Manage Files:**
    - View all uploaded documents in the sidebar
    - Click the 🗑️ button to remove documents
    - The system will automatically re-index remaining documents
    
    **4. Manage Conversations:**
    - Click "New Chat" to start a fresh conversation
    - Access recent chats from the sidebar
    
    ### 🎯 Example Questions:
    - "What is this document about?"
    - "Summarize the main points"
    - "What does the author say about [topic]?"
    - "Calculate 456 × 789"
    - "What's the capital of France?" (general knowledge)
    """)
