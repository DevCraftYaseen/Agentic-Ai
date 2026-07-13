from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

tinyllama = ChatHuggingFace(llm = llm)

result = tinyllama.invoke("Who is the most famous person in Pakistan.")

print(result.content)
