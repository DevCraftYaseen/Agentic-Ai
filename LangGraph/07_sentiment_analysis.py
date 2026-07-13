# Imports
from langgraph.graph import StateGraph , START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel , Field 

# Load ENV
load_dotenv()

# Define Schemas
class SentimentSchema(BaseModel):
    sentiment : Literal['Negative', 'Positive'] = Field(description="Sentiment of the review Either Negative Positive.")
    
class diagnoseSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')
    
class SentimentState(TypedDict):
    review : str
    sentiment : Literal['Positive', 'Negative']
    diagnosis : dict
    response : str
    
# Models
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
sentiment_model = model.with_structured_output(SentimentSchema)
diagnose_model = model.with_structured_output(diagnoseSchema)

# Functions
def get_sentiment(state: SentimentState):
    review = state['review']
    
    prompt = f"Find out the sentiment of the given review either Negative or Positive \n review -> {review}"
    
    result = sentiment_model.invoke(prompt)
    
    return {'sentiment' : result.sentiment}

def diagnose_review(state: SentimentState):
    review = state['review']
    
    prompt = f"""Diagnose this negative review:\n\n{review}\n"
    "Return issue_type, tone, and urgency. """
    
    diagnose_result = diagnose_model.invoke(prompt)
    
    return {'diagnosis' : diagnose_result.model_dump()}

def positive_response(state: SentimentState):
    review = state['review']
    
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{review}\"\n
    Also, kindly ask the user to leave feedback on our website."""
    
    positive_response = model.invoke(prompt)
    
    return {'response' : positive_response.content}

def negative_response(state: SentimentState):
    diagnosis = state['diagnosis']
    
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message. """
    
    negative_response = model.invoke(prompt)
    
    return {'response' : negative_response.content}

def check_condition(state : SentimentState) -> Literal['diagnose_review', 'positive_response'] :
    if state['sentiment'] == 'Positive':
        return 'positive_response'
    elif state['sentiment'] == "Negative":
        return 'diagnose_review'


# Define The Graph
graph = StateGraph(SentimentState)

# Add nodes
graph.add_node('get_sentiment', get_sentiment)
graph.add_node('diagnose_review', diagnose_review)
graph.add_node('positive_response', positive_response)
graph.add_node('negative_response', negative_response)

# Add edges
graph.add_edge(START, 'get_sentiment')
# Conditional Edge
graph.add_conditional_edges('get_sentiment', check_condition)

# negative side graph
graph.add_edge('diagnose_review', 'negative_response')
graph.add_edge('negative_response', END)

# Positive side graph
graph.add_edge('positive_response', END)

# Compile the graph
workflow = graph.compile()

review = 'Product was really good'

intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

result = workflow.invoke(intial_state)

print(result)