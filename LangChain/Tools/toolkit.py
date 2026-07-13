from langchain_community.tools import tool

# Tool 1 : Addition
@tool
def add(a : int , b: int) -> int:
    """Add two numbers"""
    return a + b

# Tool 2 : Multiplication
@tool
def multiply(self, num_1 : int, num_2 : int) -> int :
    """Multiply two numbers"""
    return num_1 * num_2

# Create a toolkit
class MathToolKit:
    def get_tools(self):
        return [add, multiply]
    
toolkit = MathToolKit()
tools = toolkit.get_tools()
for i in tools:
    print(i.name)