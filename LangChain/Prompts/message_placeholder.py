from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create the template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant that reply under 2 lines"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "Explain {topic} in simple terms with examples")
])

# Chat history 
chat_history = []

# Load history
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'domain':"Web Development", 'chat_history':chat_history, 'topic': "What is Hoisting"})

print(prompt)