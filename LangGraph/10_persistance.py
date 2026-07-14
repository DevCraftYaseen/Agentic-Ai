from langgraph.graph import StateGraph , START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class JokeState(TypedDict):
    topic : str
    joke : str
    explanation : str
    
def generate_joke(state: JokeState):
    topic = state['topic']
    prompt = f'Generate a joke on the topic - {topic}'
    response = llm.invoke(prompt).content
    
    return {'joke': response}

def generate_explanation(state: JokeState):
    joke = state['joke']
    prompt = f'Write an explanation for the joke - {joke}'
    response = llm.invoke(prompt).content
    
    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1 = {'configurable' : {'thread_id' : 1}}
result1 = workflow.invoke({'topic' : 'Pizza'}, config=config1)
print(result1['joke'])
print(workflow.get_state_history(config1))

config2 = {'configurable' : {'thread_id' : 2}}
result2 = workflow.invoke({'topic' : 'Pasta'}, config=config2)
print(result2['joke'])
print(workflow.get_state_history(config2))


