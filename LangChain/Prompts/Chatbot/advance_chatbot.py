# Imports -----------------------------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------

# Load ENV
load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Create the template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful health and fitness assistant that reply user queries in safe and polite manner under 2 lines"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "Answer this {query} user query ")
])

# List of chat history
chat_history = []

# Load history
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())
    
# Save history
def save_history():
    with open('chat_history.txt', 'w') as f:
        f.writelines(chat_history)


# Infinite loop until user type exit
while True:
    user_input = input("You : ").strip()
    
    if user_input == 'exit':
        chat_history.append(SystemMessage(content="Write a lovely leaving message to the user."))
        result = gemini.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        save_history()
        print(result.content)
        break
    
    prompt = chat_template.invoke({'chat_history':chat_history, 'query': user_input})
    chat_history.append(HumanMessage(content=user_input))
    
    result = gemini.invoke(prompt)
    chat_history.append(AIMessage(content=result.content))
    
    print("AI :",result.content)