from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

# Step 1 : Create a tool
@tool
def multiply(a : int, b : int ) -> int :
    """Given two numbers a and b, this tool returns their product"""
    # print('tool called by llm')
    return a * b

query = HumanMessage(content="What is 8 multiplied by 12?")
messages = [query]

# Step 2 : Binding Tool

# Step 2.1 : Create a chat Model
basic_chat_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# Step 2.2 : Bind the tool to the basic chat model and store it a new variable
model_with_tools = basic_chat_model.bind_tools([multiply])

# Step 3 : Tool Calling
result = model_with_tools.invoke(messages)

messages.append(result)

# Step 4 : Tool Execution
tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

final_result = model_with_tools.invoke(messages)

print(final_result.content)
