from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('./LangChain/Document/docs/A-Hunger-Artist-by-Franz-Kafka.pdf')

data = loader.load()

print(data)