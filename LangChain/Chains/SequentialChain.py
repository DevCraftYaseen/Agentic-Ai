from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Create a Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

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

chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({'topic' : "Agentic Ai Engineering"})

print(result)