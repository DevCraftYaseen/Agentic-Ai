from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

loader = PyMuPDFLoader('./LangChain/Projects/ChatDoc/A-Hunger-Artist-by-Franz-Kafka.pdf')

# Load ENV
load_dotenv()

# Format Documents
def format_docs(retrieved_docs):
    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return content

# Model 
llm =ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# # Create an Embedding Model
embedding_model = CohereEmbeddings(model="embed-v4.0")

# History
history = []

# Docs 
docs = loader.load()

# Create the Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Make Chunk of the pdf data
chunks = text_splitter.split_documents(docs)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a highly knowledgeable and approachable document analysis assistant. 
Your goal is to help the user extract insights from the provided reference documents. You always respond in a friendly, professional, and conversational tone.

RULES:
1. Grounding: You must base your answers strictly on the provided <context>. Do not invent or use outside knowledge.
2. Conciseness: Keep your answers clear and digestible (typically 1 to 3 sentences) while maintaining a polite tone.
3. Missing Data: If the answer cannot be confidently found in the context, do not guess. Politely inform the user. For example: "I couldn't find that specific information in the uploaded document, but I'm happy to help you look for something else!"

<context>
{context}
</context>"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

str_output_parser = StrOutputParser()
    
# Create the vector Store
vector_store = FAISS.from_documents(chunks, embedding=embedding_model)

# Retriver 
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = { 'k' : 3})

# The Optimized Chain
parallel_chain = RunnablePassthrough.assign(
    context=itemgetter('question') | retriever | RunnableLambda(format_docs)
)

main_chain = parallel_chain | prompt_template | llm | str_output_parser

while True:
    user_input = input("You : ").strip()
    history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    else:
        result = main_chain.invoke({'history': history, 'question': user_input})
        history.append(AIMessage(content=result))
        print("AI :", result)
        
print("Thank You for using our ChatDoc ❤️")