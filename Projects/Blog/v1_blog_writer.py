from __future__ import annotations
from pathlib import Path

import operator
from typing import TypedDict, Annotated, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class Task(BaseModel):
    id: str
    title: str
    description: str

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str

llm = ChatOllama(model = 'llama3.1:8b')

# Planning Node
def orchestrator(state: State):
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content = 'Create a blog plan with 5 - 7 sections on the following topic.'
            ),
            HumanMessage(
                content=f'Topic : {state['topic']}'
            )
        ]
    )
    return {'plan': plan}

# Triggering worker for each section 
def fanout(state: State):
    return [Send('worker', {'task': task, 'topic': state['topic'] ,'plan': state['plan']}) for task in state['plan'].tasks]

# Worker Node
def worker(payload: dict) -> dict:
    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    blog_title = plan.blog_title

    section_md = llm.invoke([
        SystemMessage(content='Write one clean markdown section.'),
        HumanMessage(content=(
            f'Blog: {blog_title}\n\n'
            f'Topic: {topic}\n\n'
            f'Section: {task.title}\n\n'
            f'Brief: {task.description}\n\n'
            f'Return only the section content in markdown'
        )).content.strip()
    ])

    return {
        'sections': [section_md.content]
    }

# Reducer Function
def reducer(state: State) -> dict:
    title = state['plan'].blog_title
    body = "\n\n".join(state['sections']).strip()

    final_md = f"# {title}\n\n{body}"

    # Save to Files
    filename = title.lower().replace(' ','-') + '.md'
    output_path = Path(filename)
    output_path.write_text(final_md, encoding='utf-8')

    return {
        'final': final_md
    }

# Build graph
graph = StateGraph(State)

# Add nodes
graph.add_node('orchestrator', orchestrator)
graph.add_node('worker', worker)
graph.add_node('reducer', reducer)

# Add edges
graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

app = graph.compile()

out = app.invoke({
    'topic': 'Write a blog on the self attention in LLM',
    'sections': []
})
