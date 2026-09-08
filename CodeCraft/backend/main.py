"""
FastAPI backend for CodeCraft chatbot with streaming support.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncIterator
import json
import asyncio
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from graph import app as graph_app

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="CodeCraft Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Schemas
# ============================================================
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    timestamp: str


class ThreadInfo(BaseModel):
    thread_id: str
    user_id: str
    last_message: Optional[str] = None
    timestamp: str


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
async def root():
    return {
        "message": "CodeCraft Chatbot API",
        "version": "1.0.0",
        "features": ["memory", "web_search", "streaming"]
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        # Simple config without checkpointer
        config = {"configurable": {"user_id": request.user_id}}
        
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id
        }
        
        result = graph_app.invoke(input_state, config)
        
        # Get last AI message
        ai_message = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        return ChatResponse(
            response=ai_message or "No response generated",
            thread_id=request.thread_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: ChatRequest) -> AsyncIterator[str]:
    """Stream chat responses using Server-Sent Events"""
    try:
        config = {"configurable": {"user_id": request.user_id}}
        
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id
        }
        
        # Stream events
        async for event in graph_app.astream(input_state, config):
            # Handle different event types
            for node_name, node_data in event.items():
                if node_name == "generate_response":
                    # Extract AI message
                    messages = node_data.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            # Stream token by token
                            content = msg.content
                            words = content.split()
                            for word in words:
                                data = json.dumps({
                                    "type": "token",
                                    "content": word + " ",
                                    "done": False
                                })
                                yield f"data: {data}\n\n"
                                await asyncio.sleep(0.02)  # Small delay for smooth streaming
                
                elif node_name == "search_web":
                    # Notify about search
                    data = json.dumps({
                        "type": "status",
                        "content": "🔍 Searching the web...",
                        "done": False
                    })
                    yield f"data: {data}\n\n"
        
        # Send completion signal
        data = json.dumps({
            "type": "done",
            "content": "",
            "done": True
        })
        yield f"data: {data}\n\n"
    
    except Exception as e:
        error_data = json.dumps({
            "type": "error",
            "content": str(e),
            "done": True
        })
        yield f"data: {error_data}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    return StreamingResponse(
        stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Get all threads for a user (placeholder - implement with DB query)"""
    # TODO: Implement thread retrieval from PostgreSQL
    return {
        "user_id": user_id,
        "threads": []
    }


@app.get("/memories/{user_id}")
async def get_user_memories(user_id: str):
    """Get all memories for a user"""
    try:
        from graph import get_memory_store
        store = get_memory_store()
        namespace = ("memories", user_id)
        
        memories = store.search(namespace, limit=50)
        
        return {
            "user_id": user_id,
            "memories": [
                {
                    "id": mem.key,
                    "content": mem.value.get("content", ""),
                    "timestamp": mem.value.get("timestamp", ""),
                    "source": mem.value.get("source", "")
                }
                for mem in memories
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """Delete a specific memory"""
    try:
        from graph import get_memory_store
        store = get_memory_store()
        namespace = ("memories", user_id)
        
        store.delete(namespace, memory_id)
        
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
