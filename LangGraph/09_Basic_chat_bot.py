from langgraph.graph import StateGraph , START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

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

thread_id = '1'

while True:
    user_message = input('Type here... ')
    
    print('User :', user_message)
    
    if user_message.strip().lower() in ['exit', 'quit']:
        break
    
    config = {'configurable': {'thread_id': thread_id}}
    response = chat_bot.invoke({'messages' : [HumanMessage(content=user_message)]}, config=config)
    
    print("AI :", response['messages'][-1].content)
    
print('Thank You For Using our chatbot ❤️')

    