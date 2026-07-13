# Imports -----------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

while True:
    user_input = input("You : ").strip()
    if user_input == 'exit':
        break
    else:
        result = gemini.invoke(user_input)
        print("AI :", result.content)
        
print("Thank You for using our ChatBot ❤️")