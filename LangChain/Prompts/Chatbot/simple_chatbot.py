# Imports -----------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# List of chat history
chat_history = []

# Infinite loop until user type exit
while True:
    user_input = input("You : ").strip()
    chat_history.append(user_input)
    
    if user_input == 'exit':
        break
    
    result = gemini.invoke(chat_history)
    chat_history.append(result.content)
    
    print("AI :",result.content)

print("Thanks for using our chat bot")