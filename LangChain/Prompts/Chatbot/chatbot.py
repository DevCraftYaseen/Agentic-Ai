# Imports -----------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# List of chat history
chat_history = [
    SystemMessage(content="You are a helpful assitant with sweet tone, reply each query under 2 lines.")
]

# Infinite loop until user type exit
while True:
    user_input = input("You : ").strip()
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        chat_history.append(SystemMessage(content="Write a lovely leaving message to the user."))
        result = gemini.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI :",result.content)
        break
    
    result = gemini.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    
    print("AI :",result.content)