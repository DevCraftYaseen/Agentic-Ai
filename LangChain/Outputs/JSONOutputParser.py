from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Create a Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = JsonOutputParser()

# Prompt Template 1
template = PromptTemplate(
    template = "Give me the name, age and country of a random cricketer. \n {format_instructions}",
    input_variables= [],
    partial_variables= {'format_instructions' : parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({})

print(result)

# prompt = template.format()

# result = model.invoke(prompt)

# final_json = parser.parse(result.content)

# print(final_json)



