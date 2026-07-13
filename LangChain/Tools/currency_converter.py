from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import tool
from langchain_core.tools import InjectedToolArg
from dotenv import load_dotenv
import requests
from typing import Annotated
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

@tool
def fetch_rate(base_curr: str, target_curr: str):
    """This function fetches the currency conversion factor between a given base currency and a target currency."""
    url = f"https://v6.exchangerate-api.com/v6/8f44bbca2aef5d1764f95b62/pair/{base_curr}/{target_curr}"
    response = requests.get(url)
    return response.json()

@tool
def convert(base_curr_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Given the conversion rate, this function calculates the target currency value from a given base_currency_value"""
    return base_curr_value * conversion_rate

model_with_tools = model.bind_tools([fetch_rate, convert])

query = HumanMessage(content='What is the conversion factor between USD and PKR, and based on that convert 100 USD to PKR')
messages = [query]

# 1. State variable to hold the injected argument between loops
current_conversion_rate = None 

# 2. The Agentic Decision Loop
while True:
    response = model_with_tools.invoke(messages)
    messages.append(response)
    
    # 3. The Break Condition: If the model didn't call a tool, we have our final answer!
    if not response.tool_calls:
        print("\n--- FINAL ANSWER ---")
        print(response.content[0]['text'])
        break
        
    # 4. Tool Execution Block
    for tool_call in response.tool_calls:
        print(f"Executing Tool: {tool_call['name']}")
        
        if tool_call['name'] == 'fetch_rate':
            tool_result = fetch_rate.invoke(tool_call)
            
            # Save the rate into our Python state for the next loop
            rate_data = json.loads(tool_result.content)
            current_conversion_rate = rate_data['conversion_rate']
            
            messages.append(tool_result)
            
        elif tool_call['name'] == 'convert':
            # Inject the saved state variable into the LLM's tool call arguments
            tool_call['args']['conversion_rate'] = current_conversion_rate
            
            tool_result = convert.invoke(tool_call)
            messages.append(tool_result)