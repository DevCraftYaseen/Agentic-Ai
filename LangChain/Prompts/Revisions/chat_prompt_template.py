from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ('system', 'You are a helful {domain} assitant that answers each query under 1 line in simple words.'),
    ('human', 'Answer in 1 line {topic}.')
])

prompt = template.invoke({
    'domain' : 'AI',
    'topic': 'What is ML'
})

print(prompt)