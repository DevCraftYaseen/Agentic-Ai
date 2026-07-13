from langchain_community.document_loaders import TextLoader

loader = TextLoader('./LangChain/Document/tables_1_to_100.txt', encoding = 'utf-8')

data = loader.load()

print(len(data))