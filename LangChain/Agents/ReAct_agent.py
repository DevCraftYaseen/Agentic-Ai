from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define the search tool using DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Define the LLM using ChatGoogleGenerativeAI with a specified model
llm = ChatOllama(model = 'qwen2.5-coder:3b')

# Load and pull the ReAct agent prompt from Hub
prompt = hub.pull('hwchase17/react')

# Create a ReAct Agent with the specified tools and prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# Create an AgentExecutor to manage the execution of the agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

def execute_query(query):
    try:
        # Invoke the agent executor with the user input query
        response = agent_executor.invoke({'input': query})
        return response['output']
    except Exception as e:
        # Handle any exceptions that may occur during execution
        print(f"An error occurred: {e}")
        return None

# Example usage of the execute_query function
if __name__ == "__main__":
    input_query = 'Top 3 scholarships that are fully or partially funded for masters to pakistani CS, IT or AI students from Australia.'
    result = execute_query(input_query)
    
    if result:
        print(result)
