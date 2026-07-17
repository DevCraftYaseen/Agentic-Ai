from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ThreadItem(BaseModel):
    thread_id: str
    title: str