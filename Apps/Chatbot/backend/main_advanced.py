"""
Advanced FastAPI Backend with:
- Streaming chat responses
- Tool execution tracking
- HITL approval handling
- Document upload for RAG
- Thread management with titles
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil

from advanced_graph import advanced_chat_bot, conn, llm, add_document, clear_documents
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command

app = FastAPI(title="Advanced AI Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== SCHEMAS ====================

class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


class ThreadResponse(BaseModel):
    thread_id: str
    title: str


# ==================== HELPER FUNCTIONS ====================

def get_or_create_title(thread_id: str, first_message: str) -> str:
    """Generate or retrieve a thread title."""
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    
    if row:
        return row[0]
    
    # Generate title
    try:
        prompt = f"Generate a brief, 3-5 word title for a chat starting with: '{first_message[:100]}'. Respond ONLY with the title, no quotes."
        title = llm.invoke(prompt).content.strip().strip('"\'')
    except:
        title = "New Conversation"
    
    cursor.execute(
        "INSERT INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title)
    )
    conn.commit()
    
    return title


def get_all_threads() -> List[dict]:
    """Get all threads with titles."""
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, title FROM chat_titles ORDER BY created_at DESC")
    rows = cursor.fetchall()
    return [{"thread_id": row[0], "title": row[1]} for row in rows]


def get_thread_history(thread_id: str) -> List[dict]:
    """Get chat history for a thread."""
    try:
        state = advanced_chat_bot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        
        formatted_messages = []
        for msg in messages:
            # Skip system messages and tool messages
            if msg.__class__.__name__ in ['SystemMessage', 'ToolMessage']:
                continue
            
            role = 'user' if msg.__class__.__name__ == 'HumanMessage' else 'assistant'
            
            # Handle tool calls in AI messages
            content = msg.content
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_info = []
                for tool_call in msg.tool_calls:
                    tool_info.append(f"🔧 Using tool: {tool_call['name']}")
                if tool_info and not content:
                    content = "\n".join(tool_info)
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        return formatted_messages
    except Exception as e:
        return []


# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "message": "Advanced AI Chatbot API",
        "capabilities": ["Tools", "RAG", "HITL", "Streaming"],
        "status": "online"
    }


@app.get("/api/threads", response_model=List[ThreadResponse])
def fetch_threads():
    """Get all chat threads."""
    return get_all_threads()


@app.get("/api/threads/{thread_id}")
def fetch_history(thread_id: str):
    """Get chat history for a specific thread."""
    return get_thread_history(thread_id)


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Delete a chat thread."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_titles WHERE thread_id = ?", (thread_id,))
    conn.commit()
    return {"status": "deleted", "thread_id": thread_id}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses with tool execution tracking."""
    
    # Ensure thread has a title
    get_or_create_title(request.thread_id, request.message)
    
    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}
        
        try:
            for chunk, metadata in advanced_chat_bot.stream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                # Only stream AIMessageChunk content
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    safe_content = chunk.content.replace('\n', '\\n')
                    yield f"data: {safe_content}\n\n"
                
                # Stream tool usage notifications
                elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        yield f"data: [TOOL: {tool_name}]\n\n"
                        
        except Exception as e:
            error_msg = f"Error: {str(e)}".replace('\n', '\\n')
            yield f"data: {error_msg}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat/approval")
def handle_approval(request: ApprovalRequest):
    """Handle HITL approval for stock purchases."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Resume the graph with approval decision
        result = advanced_chat_bot.invoke(
            Command(resume={'approved': 'yes' if request.approved else 'no'}),
            config=config
        )
        
        return {"status": "approved" if request.approved else "declined"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interrupts/{thread_id}")
def check_interrupts(thread_id: str):
    """Check if a thread has pending interrupts (HITL approvals)."""
    try:
        state = advanced_chat_bot.get_state(config={'configurable': {'thread_id': thread_id}})
        
        if state.next and '__interrupt__' in str(state.values):
            # Extract interrupt details
            interrupts = []
            if hasattr(state, 'tasks'):
                for task in state.tasks:
                    if hasattr(task, 'interrupts'):
                        interrupts.extend(task.interrupts)
            
            return {
                "has_interrupt": True,
                "interrupts": interrupts
            }
        
        return {"has_interrupt": False}
    except Exception as e:
        return {"has_interrupt": False, "error": str(e)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document for RAG."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to vector store
        result = add_document(file_path)
        
        return {
            "status": "success",
            "message": result,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.delete("/api/documents/clear")
def clear_all_documents():
    """Clear all uploaded documents."""
    result = clear_documents()
    return {"status": "success", "message": result}


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "llm": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
