from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Create a Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Person(BaseModel):
    name : str = Field(description='Name of the User.')
    age : int = Field(gt=18, description='Age of the User. It must be greater than 18 years.')
    country : str = Field(description='Name of the country the User belongs with.')

parser = PydanticOutputParser(pydantic_object=Person)

# Prompt Template 1
template = PromptTemplate(
    template = "Give me the name, age and country of a random cricketer. \n {format_instructions}",
    input_variables= [],
    partial_variables= {'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)




