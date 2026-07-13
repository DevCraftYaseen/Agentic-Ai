from langchain_classic.retrievers import WikipediaRetriever

# 1. Initialize the retriever
retriever = WikipediaRetriever(top_k_results=1, lang='en')

query = "Who is Imran Khan."

# 2. Invoke the search
result = retriever.invoke(query)

print(result)