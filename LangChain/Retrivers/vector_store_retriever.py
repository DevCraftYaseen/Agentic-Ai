from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load the environment variables
load_dotenv()

# Create an Embedding Model
embeddings = CohereEmbeddings(model="embed-v4.0")


documents = [
Document(page_content="LangChain helps developers build LLM applications easily."),
Document(page_content="Chroma is a vector database optimized for LLM-based search."),
Document(page_content="Embeddings convert text into high-dimensional vectors."),
Document(page_content="OpenAI provides powerful embedding models."),
]

# Create the Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

# Add Documents
vector_store.add_documents(documents=documents)

retriver = vector_store.as_retriever(search_kwargs = {"k" : 2})

results = retriver.invoke("What is LLM ?")

for i, doc in enumerate(results):
    print(f"Result Page No : {i + 1}")
    print(f"Result Content : {doc.page_content}")

