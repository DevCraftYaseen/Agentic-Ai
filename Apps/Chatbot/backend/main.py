from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schemas import ChatRequest
from services import get_or_create_title, get_all_threads, get_thread_history
from graph import chat_bot

# Updated Import: We explicitly bring in AIMessageChunk
from langchain_core.messages import HumanMessage, AIMessageChunk

app = FastAPI()

# Allow Next.js to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/threads")
def fetch_threads():
    """Returns the list of all threads with their dynamic titles."""
    return get_all_threads()

@app.get("/api/threads/{thread_id}")
def fetch_history(thread_id: str):
    """Returns the chat history for a specific thread."""
    return get_thread_history(thread_id)

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streams the LLM response back to Next.js using Server-Sent Events."""
    
    # Ensure the thread has a title before processing the chat
    get_or_create_title(request.thread_id, request.message)
    
    async def event_generator():
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Iterate through the LangGraph stream
        for chunk, metadata in chat_bot.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="messages"
        ):
            # STRICT CHECK: Only yield individual tokens, ignore the final compiled AIMessage
            if isinstance(chunk, AIMessageChunk):
                # Format as Server-Sent Event (SSE)
                # We replace newlines to prevent breaking the SSE format
                safe_content = chunk.content.replace('\n', '\\n')
                yield f"data: {safe_content}\n\n"
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")