from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Load ENV
load_dotenv()

# Model 
gemini = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# # Create an Embedding Model
embedding_model = CohereEmbeddings(model="embed-v4.0")

# Prompt template
prompt_template = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# Create the Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

str_output_parser = StrOutputParser()

# Format Documents
def format_docs(retrieved_docs):
    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return content

video_id = "Gfr50f6ZBvo"
# question = "Does the concepts of aliens discussed in the video, if yes then what was it ?"

# Fetch Transcript
try:
    raw_transcript = YouTubeTranscriptApi().fetch(video_id = video_id, languages = ['en'])
    
    transcript_list = list(raw_transcript)
    
    transcript = " ".join(chunk.text for chunk in transcript_list)
    
except TranscriptsDisabled:
    print('No Captions Available for this video')
    
chunks = text_splitter.create_documents([transcript])

# Create the vector Store
vector_store = FAISS.from_documents(chunks, embedding=embedding_model)

# Retriver 
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = { 'k' : 4})

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})

main_chain = parallel_chain | prompt_template | gemini | str_output_parser

answer = main_chain.invoke('Summarise the video in 3 lines')

print(answer)