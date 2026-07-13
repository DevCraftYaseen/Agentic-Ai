from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

huggin_face = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

result = huggin_face.embed_query("What is the meaning of Pakistan.")

print(str(result))