from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv 

# Load the environment variables
load_dotenv()

# Create an Embedding Model
embeddings = CohereEmbeddings(model="embed-v4.0")

# Load Data Using a data loader
loader = PyPDFLoader('./LangChain/Document/docs/A-Hunger-Artist-by-Franz-Kafka.pdf')
data = loader.load()

# Create the Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

# Add Documents
vector_store.add_documents(documents=data)

similar_doc = vector_store.similarity_search(
    query = 'Who is the Author of the Hunger Artist?',
    k = 1
)

print(similar_doc)

