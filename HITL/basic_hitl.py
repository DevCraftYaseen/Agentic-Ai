from langgraph.graph import StateGraph , START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

llm = ChatOllama(model='llama3.1:8b')

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
# Functions 
def chat_node(state: ChatState):

    decision = interrupt({
        'type': 'approval',
        'reason': 'Model is about to answer a question.',
        'question': state['messages'][-1].content,
        'instruction': 'Approve this Question? yes/no'
    })

    if decision['approved'] == 'yes':
        
        result = llm.invoke(state['messages'])
    
        return {'messages' : [result]}

    else:

        return {'messages' : [AIMessage(content=('Not Approved.'))]}

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

thread_id = '1234'
config = {'configurable': {'thread_id': thread_id}}

print("Welcome to the HITL Chatbot! Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_message = input('You: ')
    
    if user_message.strip().lower() in ['exit', 'quit']:
        break
    
    # First invocation: send user message and trigger interrupt
    result = chat_bot.invoke(
        {'messages': [HumanMessage(content=user_message)]}, 
        config=config
    )
    
    # Check if there's an interrupt (approval request)
    if '__interrupt__' in result:
        interrupt_data = result['__interrupt__'][0].value
        
        print(f"\n🔔 Approval Required:")
        print(f"   Reason: {interrupt_data['reason']}")
        print(f"   Question: {interrupt_data['question']}")
        print(f"   {interrupt_data['instruction']}")
        
        # Get user approval
        approval = input("Your decision (yes/no): ").strip().lower()
        
        # Resume execution with approval decision
        output = chat_bot.invoke(
            Command(resume={'approved': approval}), 
            config=config
        )
        
        # Display the AI response
        if output['messages']:
            print(f"\nAI: {output['messages'][-1].content}\n")
    else:
        # No interrupt, display response directly (shouldn't happen in this design)
        if result['messages']:
            print(f"\nAI: {result['messages'][-1].content}\n")
    
print('\nThank You For Using our chatbot ❤️')

    