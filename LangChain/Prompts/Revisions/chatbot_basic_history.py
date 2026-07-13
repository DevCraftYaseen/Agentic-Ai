# Imports -----------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

history = [
    SystemMessage(content='You are a helful math assitant that answers each question in 1 step.')
]

while True:
    user_input = input("You : ").strip()
    # history.append({"You" : user_input})
    # history.append(user_input)
    history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    else:
        result = gemini.invoke(history)
        # history.append({"AI" : result.content})
        # history.append(result.content)
        history.append(AIMessage(content=result.content))
        print("AI :", result.content)
        
print("Thank You for using our ChatBot ❤️")
# print(history)