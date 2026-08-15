from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import os

load_dotenv()

# STOCK API Key 
STOCK_API_KEY = os.getenv('STOCK_API_KEY')

# System prompt to guide the agent's behavior
SYSTEM_PROMPT = """You are a helpful stock trading assistant. You can help users with:
1. Getting real-time stock prices for any ticker symbol (e.g., AAPL, TSLA, GOOGL)
2. Simulating stock purchases after user approval
3. Answering general questions about stocks and trading

Guidelines:
- Only use the get_stock_price tool when the user asks about a specific stock price
- Only use the purchase_stock tool when the user explicitly wants to buy stocks
- For general conversation or questions, respond directly without using tools
- Always be helpful, accurate, and conversational
- If you don't know something, admit it rather than making up information

Available tools:
- get_stock_price(symbol): Get current price for a stock ticker
- purchase_stock(symbol, quantity): Simulate buying stocks (requires approval)"""

# 1. LLM with system prompt
llm = ChatOllama(model='llama3.1:8b')

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA', 'GOOGL').
    Returns the current price, change, and other market data.
    Use this when the user asks about a stock's current price.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        
        # Check for API errors
        if "Error Message" in data:
            return {"error": f"Invalid stock symbol: {symbol}"}
        elif "Note" in data:
            return {"error": "API rate limit reached. Please try again later."}
        elif "Global Quote" not in data or not data["Global Quote"]:
            return {"error": f"No data found for symbol: {symbol}"}
        
        return data
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@tool 
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
    This tool requires human approval before executing the purchase.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        quantity: Number of shares to purchase (must be positive)
    
    Returns:
        Dictionary with purchase status and details
    """
    # Validate inputs
    if quantity <= 0:
        return {
            'status': 'error',
            'message': f'Invalid quantity: {quantity}. Must be a positive number.',
            'symbol': symbol,
            'quantity': quantity
        }
    
    # HUMAN-IN-THE-LOOP: Request approval before purchase
    decision = interrupt({
        'type': 'approval',
        'reason': 'Stock purchase requires human approval',
        'symbol': symbol,
        'quantity': quantity,
        'instruction': f'Approve purchasing {quantity} shares of {symbol}? (yes/no)'
    })

    # Check if approved (decision is passed via Command(resume={...}))
    if isinstance(decision, dict) and decision.get('approved', '').strip().lower() == 'yes':
        return {
            'status': 'success',
            'message': f'✅ Successfully purchased {quantity} shares of {symbol}',
            'symbol': symbol,
            'quantity': quantity
        }
    else:
        return {
            'status': 'cancelled',
            'message': f'❌ Purchase of {quantity} shares of {symbol} was declined',
            'symbol': symbol,
            'quantity': quantity
        }
    
tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# 3. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Nodes
def chat_node(state: ChatState):
    """LLM node that may answer directly or request a tool call."""
    messages = state["messages"]
    
    # Add system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# 6. Graph
graph = StateGraph(ChatState)

# Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile Graph
chat_bot = graph.compile(checkpointer=checkpointer)

# Main loop
def main():
    thread_id = 'stock_agent_001'
    config = {'configurable': {'thread_id': thread_id}}

    print("=" * 60)
    print("🤖 Welcome to the Stock Trading Agent!")
    print("=" * 60)
    print("\nI can help you with:")
    print("  • Get real-time stock prices (e.g., 'What's the price of AAPL?')")
    print("  • Simulate stock purchases (e.g., 'Buy 10 shares of TSLA')")
    print("  • Answer questions about stocks and trading")
    print("\nType 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_message = input('You: ').strip()
        
        if not user_message:
            continue
        
        if user_message.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            break
        
        try:
            # Send user message to the agent
            result = chat_bot.invoke(
                {'messages': [HumanMessage(content=user_message)]}, 
                config=config
            )
            
            # Check if there's an interrupt (approval request for stock purchase)
            if '__interrupt__' in result:
                interrupt_data = result['__interrupt__'][0].value
                
                print(f"\n{'='*60}")
                print(f"🔔 APPROVAL REQUIRED")
                print(f"{'='*60}")
                print(f"Reason: {interrupt_data['reason']}")
                print(f"Symbol: {interrupt_data['symbol']}")
                print(f"Quantity: {interrupt_data['quantity']}")
                print(f"\n{interrupt_data['instruction']}")
                print(f"{'='*60}")
                
                # Get user approval
                approval = input("Your decision: ").strip().lower()
                
                # Resume execution with approval decision
                output = chat_bot.invoke(
                    Command(resume={'approved': approval}), 
                    config=config
                )
                
                # Display the AI response
                if output.get('messages'):
                    last_message = output['messages'][-1]
                    if hasattr(last_message, 'content'):
                        print(f"\n🤖 Agent: {last_message.content}\n")
            else:
                # No interrupt, display response directly
                if result.get('messages'):
                    last_message = result['messages'][-1]
                    if hasattr(last_message, 'content'):
                        print(f"\n🤖 Agent: {last_message.content}\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            print("Please try again or type 'exit' to quit.\n")
    
    print('\n' + '='*60)
    print('👋 Thank you for using the Stock Trading Agent!')
    print('='*60 + '\n')

if __name__ == "__main__":
    main()
