from langchain_community.tools import StructuredTool
from pydantic import BaseModel , Field

# Step 1 : Write The Schema for tool input
class input_schema(BaseModel):
    num_1 : int = Field(required = True, description= "The first number to add")
    num_2 : int = Field(required = True, description= "The second number to add")
    
# Step 2 : Write the Function
def add_nums(num_1 : int , num_2 : int) -> int :
    """Add two Numbers"""
    return num_1 + num_2

# Step 3 : Create The Structured Tool
add_nums_tool = StructuredTool.from_function(
    func= add_nums,
    name= "Addition",
    description= "Add two Numbers",
    args_schema= input_schema
)

# Call The Tool
result = add_nums_tool.invoke({
    'num_1' : 12,
    'num_2' : 4
})

# Print The Result 
print(result)

# Properties of tools
# 1. Name
print(add_nums_tool.name)

# 2. Description
print(add_nums_tool.description)

# 3. args
print(add_nums_tool.args)