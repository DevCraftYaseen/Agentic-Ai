from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

ollama_model = ChatOllama(model = 'qwen2.5-coder:3b')

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
        chain = template | ollama_model
        result = chain.invoke({'history': history, 'query': user_input})
        history.append(AIMessage(content=result.content))
        print("AI :", result.content)
        
print("Thank You for using our ChatBot ❤️")
