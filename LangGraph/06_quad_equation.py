from langgraph.graph import StateGraph , START, END
from typing import TypedDict, Literal

class QuadState(TypedDict):
    
    a: int
    b: int
    c: int
    
    equation : str
    discriminant : str
    result : str
    
# Define Functions
def show_equation(state: QuadState):
    equation = f'{state['a']}x^2 {state['b']}x {state['c']}'
    
    return {'equation' : equation}

def calculate_discriminant(state: QuadState):
    discriminant = (state['b']**2) - (4 * state['a'] * state['c'])
    
    return {'discriminant' : discriminant}

def real_roots(state: QuadState):
    root1  = (-state['b'] + state['discriminant'] ** 0.5) / (2 * state['a'])
    root2  = (-state['b'] - state['discriminant'] ** 0.5) / (2 * state['a'])
    
    result = f"The roots are {root1} and {root2}"
    
    return {'result' : result}

def repeated_root(state: QuadState):
    root  = (-state['b']) / (2 * state['a'])
    result = f"Only repeating root is {root}"
    
    return {'result' : result}

def no_real_root(state: QuadState):
    result = f"No real roots"
    
    return {'result' : result}

def check_condition(state : QuadState) -> Literal['real_roots', 'repeated_root', 'no_real_root'] :
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_root'
    else :
        return 'no_real_root'

# Define the Graph
graph = StateGraph(QuadState)

# Add Nodes to the graph
graph.add_node('show_equation',show_equation)
graph.add_node('calculate_discriminant',calculate_discriminant)
graph.add_node('real_roots',real_roots)
graph.add_node('repeated_root',repeated_root)
graph.add_node('no_real_root',no_real_root)

# Add Edges to the Graph
graph.add_edge(START, 'show_equation')
graph.add_edge("show_equation", 'calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant', check_condition)
graph.add_edge('real_roots', END)
graph.add_edge('repeated_root', END)
graph.add_edge('no_real_root', END)

workflow = graph.compile()

# Initial State
initial_state = {
    'a' : 4,
    'b' : 2,
    'c' : 2,
}

result = workflow.invoke(initial_state)

print(result)
