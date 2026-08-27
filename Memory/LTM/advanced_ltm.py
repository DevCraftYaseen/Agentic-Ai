from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Langgraph LTM Advanced'

SYSTEM_PROMPT_TEMPLATE = """You are a helpful and friendly AI assistant with memory capabilities.

User Context:
{user_details_content}

Guidelines:
- Use the user's name naturally in conversation when you know it
- Reference their interests, projects, or preferences when relevant
- Keep responses concise and conversational
- Don't over-explain or be overly formal
- Only suggest follow-up questions if truly relevant (max 2 questions)
- If no user context exists, respond naturally without mentioning missing information
- Be helpful and direct - avoid unnecessary small talk

Remember: Personalize when appropriate, but keep it natural and conversational."""

memory_llm = ChatOllama(model="llama3.1:8b", temperature=0)
chat_llm = ChatOllama(model="llama3.1:8b")
# memory_llm = ChatOllama(model="qwen3.5:9b", temperature=0)
# chat_llm = ChatOllama(model="qwen3.5:9b")

DB_URI = os.getenv('DB_URI')


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this memory is NEW and should be stored. False if duplicate/already known.")


class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")


memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """Extract user information worth remembering long-term from the message below.

Current stored memories:
{user_details_content}

Rules:
- Extract ONLY factual information: name, profession, interests, projects, preferences
- Keep each memory as a single, clear sentence
- Set is_new=true ONLY if this is genuinely NEW information not already stored
- If information is similar to existing memory, set is_new=false
- Ignore questions, greetings, or temporary statements
- If nothing worth storing, set should_write=false

Examples:
- "Hi, I'm Yaseen" → "Name: Yaseen" (is_new=true if not stored)
- "I love Python" → "Prefers Python programming language" (is_new=true)
- "How are you?" → Nothing to store (should_write=false)
- "I work on AI" when "Works as AI engineer" exists → is_new=false
"""


def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_text = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}


def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    
    if items:
        user_details = "What I know about you:\n" + "\n".join(f"- {it.value.get('data', '')}" for it in items)
    else:
        user_details = "No previous context available."

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details)
    )

    response = chat_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)


def main():
    print("🧠 Welcome to the Long-Term Memory Chatbot!")
    print("📝 Your personal details will be remembered across sessions")
    print("💾 Memories stored in PostgreSQL database")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'memories' to view your stored memories.\n")

    with PostgresStore.from_conn_string(DB_URI) as store:
        store.setup()

        graph = builder.compile(store=store)

        user_id = input("Enter your user ID (e.g., 'user-1'): ").strip() or "user-1"
        config = {"configurable": {"user_id": user_id}}

        print(f"\n👤 User ID: {user_id}")
        print("=" * 60)

        while True:
            user_message = input('\n\nYou: ').strip()

            if user_message.lower() in ['exit', 'quit', '']:
                break

            if user_message.lower() == 'memories':
                print("\n🧠 Your Stored Memories:")
                print("-" * 60)
                items = store.search(("user", user_id, "details"))
                if items:
                    for idx, item in enumerate(items, 1):
                        print(f"{idx}. {item.value.get('data', '')}")
                else:
                    print("No memories stored yet.")
                print("-" * 60)
                continue

            try:
                print('\n🤖 AI: ', end='', flush=True)

                result = graph.invoke(
                    {"messages": [{"role": "user", "content": user_message}]},
                    config
                )

                ai_response = result["messages"][-1].content
                print(ai_response)

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f'\n\n❌ Error: {str(e)}')
                print('Please try again.')

        print('\n' + '=' * 60)
        print('✅ Thank you for using the chatbot! Your memories have been saved.')
        print(f'💡 Tip: Run again with user ID "{user_id}" to continue with your memories!')


if __name__ == '__main__':
    main()
