from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Hi My name is Yaseen Khan",
    "I am 22 years old",
    "I study in 8th sem of bs it"
]

embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', output_dimensionality=32)

result = embedding_model.embed_documents(docs)

print(str(result))