from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

docs = [
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli."

embedding_model = GoogleGenerativeAIEmbeddings(model='text-embedding-004', output_dimensionality=768)

# Embed Docs 
embedded_docs = embedding_model.embed_documents(docs)

# Embed Query
embedded_query = embedding_model.embed_query(query)

print("Total docs in list:", len(docs))
print("Total embeddings generated:", len(embedded_docs))

print("1. Type of embedded_docs:", type(embedded_docs))
print("2. Length of embedded_docs:", len(embedded_docs))

if len(embedded_docs) > 0:
    print("3. Type of first item:", type(embedded_docs[0]))
    print("4. Length of first item:", len(embedded_docs[0]))
    
    # Let's peek at the first few numbers of whatever is inside
    if isinstance(embedded_docs[0], list):
        print("5. Peek inside:", embedded_docs[0][:3])

# 1. Calculate similarity
# scores = cosine_similarity([embedded_query], embedded_docs)

# 2. Extract the 1D list of scores from the 2D matrix
# flat_scores = scores[0] 
# print(f"All Scores: {flat_scores}")

# # 3. Get the index of the maximum score using NumPy
# most_similar_index = np.argmax(flat_scores)
# print(f"\nHighest Score Index: {most_similar_index}")

# # 4. Use that index to get the actual document
# best_match = docs[most_similar_index]
# print(f"Best matching document: {best_match}")