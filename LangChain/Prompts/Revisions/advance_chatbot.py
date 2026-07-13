from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Chat Prompt Template
template = ChatPromptTemplate([
    ('system', 'You are a helful Ai assistant that responds to every query under 1 line.'),
    MessagesPlaceholder(variable_name = 'history'),
    ('human', '{query}')
])

history = [
    HumanMessage(content = 'What is ML?'),
    AIMessage(content = 'ML is learning patterns from data.')
]

while True:
    user_input = input("You : ").strip()
    history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    else:
        prompt = template.invoke({'history': history, 'query': user_input})
        result = gemini.invoke(prompt)
        history.append(AIMessage(content=result.content))
        print("AI :", result.content)
        
print("Thank You for using our ChatBot ❤️")
    