from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, SystemMessage
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

load_dotenv()

llm = ChatOllama(model = 'qwen3.5:9b')

# Create store 
store = InMemoryStore()
user_id = 'user-1'

# Create namespace
user_details = ('users', user_id, 'details')

# Add memories
store.put(user_details, "profile_1", {"data": "Name: Yaseen"})
store.put(user_details, "profile_2", {"data": "Profession: Teaches AI on LinkedIn"})
store.put(user_details, "preference_1", {"data": "Prefers concise answers"})
store.put(user_details, "preference_2", {"data": "Likes examples in Python"})
store.put(user_details, "project_1", {"data": "Building MCP servers (Python-based project)"})

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
say "Since your project is built with TypeScript..."

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user’s memory (which may be empty) is provided as: {user_detailed_content}
"""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config: RunnableConfig, store: BaseStore):
    user_id = config['configurable']['user_id']
    
    user_details = ('users', user_id, 'details')
    memories = store.search(user_details)

    if memories:
        user_detailed_content = '\n'.join(f"- {memory.value.get('data', '')}" for memory in memories)
    else:
        user_detailed_content = ''

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_detailed_content= user_detailed_content
    )

    system_message = SystemMessage(content=system_prompt)

    response = llm.invoke([system_message] + state['messages'])

    return {'messages': [response]}


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

chatbot = builder.compile(store=store)

config = {"configurable": {"user_id": "user-1"}}

result = chatbot.invoke(
    {"messages": [{"role": "user", "content": "Explain gen ai in simple terms."}]},
    config,
)

print(result["messages"][-1].content)