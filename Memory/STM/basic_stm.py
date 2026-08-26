import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
import sys

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Langgraph chatbot STM'

DB_URI = os.getenv('DB_URI')

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
    print("🤖 Welcome to the Short-Term Memory Chatbot with Streaming!")
    print("📝 Your conversation will be saved in PostgreSQL database")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        chat_bot = graph.compile(checkpointer=checkpointer)

        thread_id = 'terminal-session-1'
        config = {'configurable': {'thread_id': thread_id}}

        print(f"💾 Session ID: {thread_id}")
        print("=" * 60)

        while True:
            user_message = input('\n\nYou: ').strip()

            if user_message.lower() in ['exit', 'quit', '']:
                break

            try:
                print('\n🤖 AI: ', end='', flush=True)

                full_response = ""

                async for chunk, metadata in chat_bot.astream(
                    {'messages': [HumanMessage(content=user_message)]},
                    config=config,
                    stream_mode='messages'
                ):
                    if isinstance(chunk, AIMessageChunk):
                        if chunk.content:
                            print(chunk.content, end='', flush=True)
                            full_response += chunk.content

                print()

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f'\n\n❌ Error: {str(e)}')
                print('Please try again.')

        print('\n' + '=' * 60)
        print('✅ Thank you for using the chatbot! Your conversation has been saved.')
        print('💡 Tip: Run again with the same thread_id to continue your conversation!')


if __name__ == '__main__':
    if sys.platform == 'win32':
        import selectors
        asyncio.run(
            main(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        )
    else:
        asyncio.run(main())
