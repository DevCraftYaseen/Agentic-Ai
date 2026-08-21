"""
Advanced FastAPI Backend with:
- Streaming chat responses
- Tool execution tracking
- HITL approval handling
- Document upload for RAG
- Thread management with titles
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil

from graph import graph, conn, llm_conversational, add_document, clear_documents, vector_store, VECTOR_STORE_PATH
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Set once the lifespan startup below finishes compiling the graph with its
# async checkpointer. Endpoints reference this module-level name at call
# time (not import time), so it's safe for them to be defined before this
# is assigned — FastAPI guarantees lifespan startup completes before any
# request is served.
chat_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_bot
    # Deliberately a separate db file from conn's chatbot.db (chat_titles
    # table) — two different sqlite drivers (sync sqlite3 + aiosqlite)
    # writing to the same file concurrently risks SQLITE_BUSY lock errors.
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        chat_bot = graph.compile(checkpointer=checkpointer)
        yield
    # AsyncSqliteSaver's connection is closed automatically on exit here


app = FastAPI(title="Advanced AI Chatbot API", lifespan=lifespan)

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

    try:
        prompt = f"Generate a brief, 3-5 word title for a chat starting with: '{first_message[:100]}'. Respond ONLY with the title, no quotes."
        # Use the conversational model for this — it's just a cheap
        # freeform text task, no tools/grounding needed.
        title = llm_conversational.invoke(prompt).content.strip().strip('"\'')
    except Exception:
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


async def get_thread_history(thread_id: str) -> List[dict]:
    """Get chat history for a thread."""
    try:
        state = await chat_bot.aget_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])

        formatted_messages = []
        for msg in messages:
            if msg.__class__.__name__ in ['SystemMessage', 'ToolMessage']:
                continue

            role = 'user' if msg.__class__.__name__ == 'HumanMessage' else 'assistant'

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


async def get_pending_interrupts(thread_id: str) -> List[dict]:
    """
    Check whether a thread's graph is paused on an interrupt() call and, if so,
    return the interrupt payload(s) in a plain-dict shape the frontend expects
    (`{"value": {...}}`), regardless of how LangGraph internally represents
    the Interrupt object across versions.
    """
    state = await chat_bot.aget_state(config={'configurable': {'thread_id': thread_id}})

    # state.next is a non-empty tuple (e.g. ('tools',)) when the graph is
    # paused mid-execution — this, not state.values, is what tells us
    # there's a pending interrupt.
    if not state.next:
        return []

    interrupts_data = []
    for task in state.tasks:
        task_interrupts = getattr(task, 'interrupts', None)
        if not task_interrupts:
            continue
        for intr in task_interrupts:
            # Explicitly extract .value into a plain dict so FastAPI's JSON
            # serialization can't turn this into an array the frontend can't read.
            interrupts_data.append({"value": intr.value})

    return interrupts_data


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
async def fetch_history(thread_id: str):
    """Get chat history for a specific thread."""
    return await get_thread_history(thread_id)


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

    get_or_create_title(request.thread_id, request.message)

    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}

        try:
            async for chunk, metadata in chat_bot.astream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    safe_content = chunk.content.replace('\n', '\\n')
                    yield f"data: {safe_content}\n\n"

                elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        yield f"data: [TOOL: {tool_name}]\n\n"

        except Exception as e:
            error_msg = f"Error: {str(e)}".replace('\n', '\\n')
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat/approval")
async def handle_approval(request: ApprovalRequest):
    """Handle HITL approval for stock purchases."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        await chat_bot.ainvoke(
            Command(resume={'approved': 'yes' if request.approved else 'no'}),
            config=config
        )

        return {"status": "approved" if request.approved else "declined"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interrupts/{thread_id}")
async def check_interrupts(thread_id: str):
    """Check if a thread has pending interrupts (HITL approvals)."""
    try:
        interrupts_data = await get_pending_interrupts(thread_id)
        return {
            "has_interrupt": len(interrupts_data) > 0,
            "interrupts": interrupts_data
        }
    except Exception as e:
        return {"has_interrupt": False, "error": str(e)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document for RAG."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = add_document(file_path)

        return {
            "status": "success",
            "message": result,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/api/documents/list")
def list_documents():
    """List all uploaded documents in the vector store."""
    try:
        if not vector_store or not os.path.exists(VECTOR_STORE_PATH):
            return {"documents": []}

        docs_metadata = []

        tracking_file = os.path.join(VECTOR_STORE_PATH, "documents.json")
        if os.path.exists(tracking_file):
            import json
            with open(tracking_file, 'r') as f:
                docs_metadata = json.load(f)

        return {"documents": docs_metadata}
    except Exception as e:
        return {"documents": [], "error": str(e)}


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