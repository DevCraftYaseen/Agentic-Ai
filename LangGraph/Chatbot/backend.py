from langgraph.graph import StateGraph , START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv()

# Custom project name using code
os.environ['LANGCHAIN_PROJECT'] = 'Langgraph chatbot'


llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
# Functions 
def chat_node(state: ChatState):
    messages = state['messages']
    
    result = llm.invoke(messages)
    
    return {'messages' : [result]}

# Define Check Pointer
checkpointer = MemorySaver()

# Define graph
graph = StateGraph(ChatState)

# Add node
graph.add_node('chat_node', chat_node)

# Add edges 
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chat_bot = graph.compile(checkpointer=checkpointer)
