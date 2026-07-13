from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Create a Model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Sentiment(BaseModel):
    sentiment : Literal['Positive', 'Negative'] = Field(description='Sentiment of the review. Either Positive or Negative')

pydantic_parser = PydanticOutputParser(pydantic_object=Sentiment)

str_parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template='Classify the given review as either Positive or Negative. \n {review} \n {format_instructions}',
    input_variables= ['review'],
    partial_variables={'format_instructions' : pydantic_parser.get_format_instructions()}
)

prompt_2 = PromptTemplate(
    template='Write a single appropriate response to this positive review that i can reply to the client. \n {review}',
    input_variables= ['review']
)

prompt_3 = PromptTemplate(
    template='Write two appropriate responses to this negative review. One i want to reply to the customer and the second to my customer support team. \n {review}',
    input_variables= ['review']
)

classifier_chain = prompt_1 | model | pydantic_parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt_2 | model | str_parser),
    (lambda x:x.sentiment == 'Negative', prompt_3 | model | str_parser),
    RunnableLambda(lambda x: 'Could not Find Sentiment.')
)

main_chain = classifier_chain | branch_chain

result = main_chain.invoke({'review' : 'This is a very good smartphone.'})

print(result)