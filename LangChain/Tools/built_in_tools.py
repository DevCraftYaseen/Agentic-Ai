from langchain_community.tools import DuckDuckGoSearchRun , ShellTool

search_tool = DuckDuckGoSearchRun()

shell_tool = ShellTool()

result = search_tool.invoke("Latest Model of Iphone")

print(result)