"""
Streamlit frontend for CodeCraft Chatbot
Features:
- Real-time streaming responses
- Memory management
- Multiple conversation threads
- Web search indicator
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import List, Dict

# ============================================================
# Configuration
# ============================================================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CodeCraft Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# Session State Initialization
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"

if "threads" not in st.session_state:
    st.session_state.threads = {
        st.session_state.thread_id: {
            "messages": [],
            "title": "New Chat",
            "created": datetime.now().isoformat()
        }
    }


# ============================================================
# Helper Functions
# ============================================================
def send_message_stream(message: str, thread_id: str, user_id: str):
    """Send message and stream response"""
    try:
        response = requests.post(
            f"{API_URL}/chat/stream",
            json={
                "message": message,
                "thread_id": thread_id,
                "user_id": user_id
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code == 200:
            full_response = ""
            placeholder = st.empty()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_str)
                            event_type = data.get('type')
                            content = data.get('content', '')
                            
                            if event_type == 'token':
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                            elif event_type == 'status':
                                placeholder.info(content)
                            elif event_type == 'done':
                                placeholder.markdown(full_response)
                                return full_response
                            elif event_type == 'error':
                                placeholder.error(f"Error: {content}")
                                return None
                        except json.JSONDecodeError:
                            continue
            
            return full_response
        else:
            st.error(f"Error: {response.status_code}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def send_message(message: str, thread_id: str, user_id: str):
    """Send message without streaming (fallback)"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "thread_id": thread_id,
                "user_id": user_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response")
        else:
            st.error(f"Error: {response.status_code}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def get_memories(user_id: str) -> List[Dict]:
    """Fetch user memories"""
    try:
        response = requests.get(f"{API_URL}/memories/{user_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("memories", [])
        return []
    except:
        return []


def delete_memory(user_id: str, memory_id: str):
    """Delete a specific memory"""
    try:
        response = requests.delete(f"{API_URL}/memories/{user_id}/{memory_id}")
        return response.status_code == 200
    except:
        return False


def create_new_thread():
    """Create a new conversation thread"""
    new_thread_id = str(uuid.uuid4())
    st.session_state.thread_id = new_thread_id
    st.session_state.threads[new_thread_id] = {
        "messages": [],
        "title": "New Chat",
        "created": datetime.now().isoformat()
    }
    st.session_state.messages = []
    st.rerun()


def switch_thread(thread_id: str):
    """Switch to a different thread"""
    st.session_state.thread_id = thread_id
    st.session_state.messages = st.session_state.threads[thread_id]["messages"]
    st.rerun()


# ============================================================
# UI Layout
# ============================================================

# Sidebar
with st.sidebar:
    st.title("🤖 CodeCraft Chat")
    
    st.divider()
    
    # User settings
    st.subheader("User Settings")
    user_id = st.text_input("User ID", value=st.session_state.user_id, key="user_id_input")
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
    
    st.divider()
    
    # Conversation threads
    st.subheader("Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_thread()
    
    st.caption(f"**Current:** {st.session_state.thread_id[:8]}...")
    
    # List threads
    for thread_id, thread_data in st.session_state.threads.items():
        is_current = thread_id == st.session_state.thread_id
        button_label = f"{'▶ ' if is_current else ''}{thread_data['title'][:20]}"
        
        if st.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
            if not is_current:
                switch_thread(thread_id)
    
    st.divider()
    
    # Memory management
    st.subheader("💾 Memory")
    
    if st.button("View Memories", use_container_width=True):
        st.session_state.show_memories = not st.session_state.get("show_memories", False)
    
    if st.session_state.get("show_memories", False):
        memories = get_memories(st.session_state.user_id)
        if memories:
            st.caption(f"Found {len(memories)} memories")
            for mem in memories[:10]:
                with st.expander(mem["content"][:40] + "..."):
                    st.text(mem["content"])
                    st.caption(f"📅 {mem['timestamp'][:10]}")
                    if st.button("Delete", key=f"del_{mem['id']}"):
                        if delete_memory(st.session_state.user_id, mem["id"]):
                            st.success("Deleted!")
                            st.rerun()
        else:
            st.info("No memories yet")
    
    st.divider()
    
    # Info
    st.caption("**Features:**")
    st.caption("✓ Persistent memory")
    st.caption("✓ Web search")
    st.caption("✓ Streaming responses")
    st.caption("✓ Multiple threads")


# Main chat area
st.title("💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update thread title (first message)
    if len(st.session_state.messages) == 1:
        st.session_state.threads[st.session_state.thread_id]["title"] = prompt[:30]
    
    # Save to thread
    st.session_state.threads[st.session_state.thread_id]["messages"] = st.session_state.messages
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message_stream(
                prompt,
                st.session_state.thread_id,
                st.session_state.user_id
            )
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.threads[st.session_state.thread_id]["messages"] = st.session_state.messages


# Status bar
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🆔 Thread: {st.session_state.thread_id[:8]}")
with col2:
    st.caption(f"👤 User: {st.session_state.user_id}")
with col3:
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.caption("🟢 Backend: Online")
        else:
            st.caption("🔴 Backend: Error")
    except:
        st.caption("🔴 Backend: Offline")
