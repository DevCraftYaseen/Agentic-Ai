from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Custom project name using code
os.environ['LANGCHAIN_PROJECT'] = 'Sequencial App'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model_1 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature = 0.7)
model_2 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature = 0.5)

parser = StrOutputParser()

chain = prompt1 | model_1 | parser | prompt2 | model_2 | parser

config = {
    'run_name' : 'Sequence',
    'tags' : ['llm app', 'sequencial app', 'dcy'],
    'metadata' : {'Developer' :'devcraftyaseen'}
}

result = chain.invoke({'topic': 'Unemployment in Pakistan'}, config=config)

print(result)
