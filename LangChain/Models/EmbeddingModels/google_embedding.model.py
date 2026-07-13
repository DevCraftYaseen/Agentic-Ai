from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

docs = [
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli."

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",
    output_dimensionality=768,)

# Emebed Docs 
embedded_docs = embedding_model.embed_documents(docs)

# Emebed Query
embedded_query = embedding_model.embed_query(query)

# print(embedded_docs)
# print()
# print(embedded_query)

scores = cosine_similarity([embedded_query], embedded_docs)
print(scores)

