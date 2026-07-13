from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= './LangChain/Document/docs',
    glob= '*.pdf',
    loader_cls=PyPDFLoader
)

# lazy Load
data = loader.lazy_load()

for document in data:
    print(document.metadata)
    