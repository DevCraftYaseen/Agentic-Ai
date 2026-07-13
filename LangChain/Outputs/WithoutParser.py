from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

deepseekv4 = ChatHuggingFace(llm = llm)

# Prompt Template 1
template_1 = PromptTemplate(
    template = "Write a detailed report on the topic {topic}",
    input_variables= ['topic']
)

# Prompt Template 2
template_2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. /n {text}",
    input_variables= ['text']
)

# Prompt 1
prompt_1 = template_1.invoke({'topic' : 'Machine Learning'})

result_1 = deepseekv4.invoke(prompt_1)

# Prompt 2
prompt_2 = template_2.invoke({'text': result_1.content})

result_2 = deepseekv4.invoke(prompt_2)

print(result_2.content)