# Imports
from langgraph.graph import StateGraph , START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Define a model
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# Define State
class BlogState(TypedDict):
    topic : str
    outline : str
    content : str


# Define Functions
# 1. Generate Outline
def generate_outline(state : BlogState) -> BlogState:
    topic = state['topic']
    
    prompt = f'Generate a detailed outline for a blog on the topic - {topic}.'
    
    result = model.invoke(prompt)
    
    state['outline'] = result.content
    
    return state

# 2. Generate Blog
def generate_blog(state : BlogState) -> BlogState:
    topic = state['topic']
    outline = state['outline']
    
    prompt = f'Write a detailed blog on the topic - {topic} using the following outline \n {outline}.'
    
    result = model.invoke(prompt)
    
    state['content'] = result.content
    
    return state# Define Functions
# 1. Generate Outline
def generate_outline(state : BlogState) -> BlogState:
    topic = state['topic']
    
    prompt = f'Generate a detailed outline for a blog on the topic - {topic}.'
    
    result = model.invoke(prompt)
    
    state['outline'] = result.content
    
    return state

# 2. Generate Blog
def generate_blog(state : BlogState) -> BlogState:
    topic = state['topic']
    outline = state['outline']
    
    prompt = f'Write a detailed blog on the topic - {topic} using the following outline \n {outline}.'
    
    result = model.invoke(prompt)
    
    state['content'] = result.content
    
    return state

# Define the Graph
graph = StateGraph(BlogState)

# Add nodes 
graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_blog', generate_blog)

# Add edges
graph.add_edge(START, 'generate_outline')
graph.add_edge("generate_outline", 'generate_blog')
graph.add_edge('generate_blog', END)

# compile the graph
workflow = graph.compile()

# Execute Workflow
initial_state = {'topic': 'Impact of AI on Software Engineering.'}

final_state = workflow.invoke(initial_state)

print("-" * 10 , 'Blog Outline' , '-' * 10)
print(final_state['outline'])
print()
print("-" * 10 , 'Blog Content' , '-' * 10)
print(final_state['content'])
