from langchain_community.tools import tool

# Step 1 - Create A Function
def add_nums(a,b):
    """Add two numbers"""
    return a + b

# Step 2 - Add Type hints
def add_nums(a : int , b: int) -> int :
    """Add two numbers"""
    return a + b

# Step 3 - Add the @tool decorator
@tool
def add_nums(a : int , b: int) -> int:
    """Add two numbers"""
    return a + b


# Call The Tool
result = add_nums.invoke({
    'a' : 12,
    'b' : 4
})

# Print The Result 
print(result)

# Properties of tools
# 1. Name
print(add_nums.name)

# 2. Description
print(add_nums.description)

# 3. args
print(add_nums.args)