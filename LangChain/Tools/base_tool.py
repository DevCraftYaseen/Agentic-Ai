from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# Step 1 : Write The Schema for tool input
class input_schema(BaseModel):
    num_1 : int = Field(required = True, description= "The first number to multiply")
    num_2 : int = Field(required = True, description= "The second number to multiply")
    
# Step 2 : Create the Custom Tool Class by inheriting the Base Tool
class MultiplyTool(BaseTool):
    name : str = "Multiply"
    description : str = "Multiply Two Numbers"
    
    args_schema : Type[BaseModel] = input_schema
    
    def _run(self, num_1 : int, num_2 : int) -> int :
        return num_1 * num_2
    
# Step 3 : Create an Object of the MultiplyTool Class
multiply_tool = MultiplyTool()
    
# Call The Tool
result = multiply_tool.invoke({
    'num_1' : 12,
    'num_2' : 4
})

# Print The Result 
print(result)

# Properties of tools
# 1. Name
print(multiply_tool.name)

# 2. Description
print(multiply_tool.description)

# 3. args
print(multiply_tool.args)