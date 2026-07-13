from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever

# Load ENV
load_dotenv()

all_docs = [
Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Model 
gemini = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# Create an Embedding Model
embeddings = CohereEmbeddings(model="embed-v4.0")

# Create the Vector Store
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

# Add Documents
vector_store.add_documents(documents=all_docs)

base_retriver = vector_store.as_retriever(search_kwargs = {"k" : 3})

multi_query_retriver = MultiQueryRetriever.from_llm(
    retriever = base_retriver,
    llm = gemini
)

query = 'How to improve energy levels and maintain balance ?'

results = multi_query_retriver.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

