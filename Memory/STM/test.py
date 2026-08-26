"""
Run directly: python test_astream_events.py
Tests astream_events(version="v2") instead of stream_mode="messages"/"custom".
This is a different, older LangChain API (predates LangGraph's stream_mode
wrapper) that many production streaming setups rely on specifically because
it's more battle-tested. Worth isolating since stream_mode has failed
identically across two different approaches so far.
"""
import asyncio
import sys
import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as _Saver

llm = ChatOllama(model='llama3.1:8b')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def chat_node(state: ChatState, config: RunnableConfig):
    messages = state['messages']
    full = None
    async for chunk in llm.astream(messages, config=config):
        full = chunk if full is None else full + chunk
    return {'messages': [full]}


graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


async def main():
    checkpointer = _Saver()
    chat_bot = graph.compile(checkpointer=checkpointer)
    config = {'configurable': {'thread_id': 'isolation-test'}}

    print("astream_events(version='v2') test — watch the timestamps:\n")

    start = time.time()
    count = 0

    async for event in chat_bot.astream_events(
        {'messages': [HumanMessage(content="Write a 150 word paragraph about the ocean.")]},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                count += 1
                elapsed = time.time() - start
                print(f"[{elapsed:6.3f}s] chunk #{count}: {chunk.content!r}")

    total_time = time.time() - start
    print(f"\nTotal content chunks: {count}")
    print(f"Total time: {total_time:.2f}s")

    if count <= 2:
        print("\n⚠️  Still buffered even with astream_events — this points to")
        print("    something more fundamental (Ollama's async client, or the")
        print("    installed langchain-ollama version's callback support).")
    else:
        print(f"\n✅ {count} chunks streamed live via astream_events!")
        print("   Use this API instead of stream_mode in stm.py.")


if __name__ == '__main__':
    if sys.platform == 'win32':
        import selectors
        asyncio.run(
            main(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        )
    else:
        asyncio.run(main())