import sqlite3
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', streaming=True)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    result = llm.invoke(messages)
    return {'messages': [result]}

# Database connection
conn = sqlite3.connect('chatbot.db', check_same_thread=False)

# Create custom table for dynamic titles
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT
    )
''')
conn.commit()

# Initialize LangGraph Checkpointer
checkpointer = SqliteSaver(conn=conn)

# Build Graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chat_bot = graph.compile(checkpointer=checkpointer)