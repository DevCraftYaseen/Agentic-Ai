from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature=0, max_output_tokens = 200)

result = gemini.invoke("Who is the most famous celebrity of Pakisan.")

print(result.content)
