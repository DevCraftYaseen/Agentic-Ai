# Parallel Workflow in LangGraph"
# Imports
from langgraph.graph import StateGraph , START, END
from typing import TypedDict

# Define State Schema
class BatStat(TypedDict):
    # Input Types
    runs : int
    balls : int
    sixes : int
    fours : int
    
    # Output Types
    strike_rate : float
    boundary_percentage : float
    balls_per_boundary : float
    summary : str
    
# Define Functions
def calculate_strike_rate(state: BatStat):
    strike_rate = (state['runs']/state['balls']) * 100
    return {'strike_rate' : strike_rate}
    
def calculate_balls_per_boundary(state: BatStat):
    balls_per_boundary = state['balls'] / (state['fours'] + state['sixes'])
    return {'balls_per_boundary' : balls_per_boundary}

def calculate_boundary_percentage(state: BatStat):
    boundary_percentage = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    
    return {'boundary_percentage' : boundary_percentage}
    
def summary(state: BatStat):
    summary = f""""
    Batsman Strike Rate : {state['strike_rate']}
    Balls per boundary : {state['balls_per_boundary']}
    Boundary percentage : {state['boundary_percentage']}
    """
    return {'summary' : summary}

# Define the Graph
graph = StateGraph(BatStat)

# Add Nodes to the graph
graph.add_node('calculate_strike_rate',calculate_strike_rate)
graph.add_node('calculate_balls_per_boundary',calculate_balls_per_boundary)
graph.add_node('calculate_boundary_percentage',calculate_boundary_percentage)
graph.add_node('summary',summary)

# Add Edges to the Graph
graph.add_edge(START, 'calculate_strike_rate')
graph.add_edge(START, 'calculate_balls_per_boundary')
graph.add_edge(START, 'calculate_boundary_percentage')

graph.add_edge('calculate_strike_rate', 'summary')
graph.add_edge('calculate_balls_per_boundary', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

# Initial State
initial_state = {
    'runs' : 100,
    'balls' : 49,
    'fours' : 10,
    'sixes' : 6,
}

result = workflow.invoke(initial_state)

print(result)




 