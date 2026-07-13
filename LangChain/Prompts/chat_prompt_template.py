from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} assistant that reply under 2 lines"),
    ('human', "Explain {topic} in simple terms with examples")
])

prompt = chat_template.invoke({'domain':"Web Development", 'topic': "What is Hoisting"})

print(prompt)