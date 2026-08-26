from langchain_ollama import OllamaEmbeddings
from langgraph.store.memory import InMemoryStore

# Create mebeddings model
embeddings = OllamaEmbeddings(model = 'nomic-embed-text')

# Create store with semantic search capability using embeddings model
store = InMemoryStore(index={'embed' : embeddings, 'dims': 768})

# Create a namespace
namespace_1 = ('users', 'u1')

# Add memories
store.put(namespace_1, "1", {"data": "User prefers concise answers over long explanations"})
store.put (namespace_1, "2", {"data": "User likes examples in Python"})
store.put (namespace_1, "3", {"data": "User usually works late at night"})
store.put (namespace_1, "4", {"data": "User prefers dark mode in applications"})
store.put (namespace_1, "5", {"data": "User is learning machine learning"})
store.put(namespace_1, "6", {"data": "User dislikes overly theoretical expladations"})
store.put (namespace_1, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace_1, "8", {"data": "User is based in India"})
store.put (namespace_1, "9", {"data": "User likes real-world analogies"})
store.put (namespace_1, "10", {"data": "User prefers bullet points over paragraphs"})

items = store.search(namespace_1, query='What are user"s preferences?', limit=3)

for memory in items:
    print('-',memory.value['data'])

