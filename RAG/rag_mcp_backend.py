from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import asyncio
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# -------------------
# 1. LLM Configuration
# -------------------
# llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
llm = ChatOllama(model = 'llama3.1:8b')

# -------------------
# 2. RAG Components
# -------------------

# Embedding Model
# Using v3.0 (the correct model name for Cohere)
# embedding_model = CohereEmbeddings(model="embed-english-v3.0")
embedding_model = OllamaEmbeddings(model = 'nomic-embed-text')

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Smaller chunks = more precise retrieval
    chunk_overlap=200,  # Overlap helps maintain context
    separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting
)

# Document path
PDF_PATH = './LangChain/Document/docs/A-Hunger-Artist-by-Franz-Kafka.pdf'
VECTOR_STORE_PATH = './RAG/faiss_index'


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a readable string for the LLM.
    
    Why: Document objects contain metadata that LLMs don't need.
    We extract just the text content with clear separators.
    """
    if not docs:
        return "No relevant documents found."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Include page number if available
        page = doc.metadata.get('page', 'unknown')
        formatted.append(f"[Document {i} - Page {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def load_and_index_document():
    """
    Load PDF, split into chunks, and create FAISS vector store.
    
    Why: This is expensive (API calls for embeddings), so we:
    1. Check if index exists (avoid re-indexing)
    2. Save the index for reuse
    3. Handle errors gracefully
    """
    try:
        # Check if we already have an indexed vector store
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"✅ Loading existing vector store from {VECTOR_STORE_PATH}")
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            return vector_store
        
        # Load PDF
        print(f"📄 Loading document from {PDF_PATH}")
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content loaded from {PDF_PATH}")
        
        print(f"📄 Loaded {len(documents)} pages")
        
        # Split into chunks using split_documents() that handles Document objects correctly
        chunks = text_splitter.split_documents(documents)
        print(f"✂️  Created {len(chunks)} chunks")
        
        # Create vector store
        print("🔮 Creating embeddings and vector store (this may take a moment)...")
        vector_store = FAISS.from_documents(chunks, embedding=embedding_model)
        
        # Save for future use
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"💾 Vector store saved to {VECTOR_STORE_PATH}")
        
        return vector_store
    
    except FileNotFoundError:
        print(f"❌ Error: PDF file not found at {PDF_PATH}")
        raise
    except Exception as e:
        print(f"❌ Error during document indexing: {e}")
        raise


# Initialize vector store and retriever
try:
    vector_store = load_and_index_document()
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4}  # Retrieve top 4 most relevant chunks
    )
    print("✅ RAG system initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize RAG system: {e}")
    retriever = None

# -------------------
# 3. Tools
# -------------------

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div, mod
    
    Args:
        first_num: First number
        second_num: Second number
        operation: One of 'add', 'sub', 'mul', 'div', 'mod'
    """
    try:
        operation = operation.lower()
        
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        elif operation == "mod":
            if second_num == 0:
                return {"error": "Modulus by zero is not allowed"}
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation '{operation}'. Use: add, sub, mul, div, mod"}
        
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
            "formatted": f"{first_num} {operation} {second_num} = {result}"
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def search_document(query: str) -> str:
    """
    Search the uploaded document for information relevant to the query.
    Use this when the user asks questions about the document content.
    
    Args:
        query: The search query or question about the document
        
    Returns:
        Relevant excerpts from the document
    """
    if retriever is None:
        return "Error: Document retrieval system is not available. Please check if the PDF was loaded correctly."
    
    try:
        # Format documents as readable text, not raw objects
        retrieved_docs = retriever.invoke(query)
        formatted_context = format_docs(retrieved_docs)
        return formatted_context
    
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


# Bind tools to LLM
tools = [calculator, search_document]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Graph
# -------------------
def build_graph():
    """
    Build the LangGraph chatbot with RAG and calculator tools.
    
    Architecture:
    1. chat_node: LLM decides if it needs tools or can answer directly
    2. tools node: Executes the selected tool(s)
    3. Loop back to chat_node: LLM uses tool results to formulate final answer
    """
    
    async def chat_node(state: ChatState):
        """
        LLM node that processes messages and decides on tool usage.
        
        Why async: Allows concurrent operations and non-blocking I/O
        """
        messages = state["messages"]
        
        # Add system message with RAG instructions
        # This tells the LLM when and how to use the search_document tool
        system_message = SystemMessage(
            content="""You are a helpful AI assistant that can chat about any topic.

You have access to specialized tools:
- 'search_document' tool: Searches a document about "A Hunger Artist" by Franz Kafka
- 'calculator' tool: Performs math operations

IMPORTANT INSTRUCTIONS:
- For questions about the "Hunger Artist" document, story, characters, or author → Use 'search_document' tool
- For math calculations → Use 'calculator' tool  
- For ANY other topic (general knowledge, advice, conversation) → Respond naturally WITHOUT using tools

Be friendly, helpful, and conversational. You can discuss any topic the user wants!"""
        )
        
        # Insert system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_message] + messages
        
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Tool execution node
    tool_node = ToolNode(tools)

    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Define edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node",
        tools_condition  # Automatically routes to tools if LLM requests them
    )
    graph.add_edge('tools', 'chat_node')  # After tools, go back to LLM

    # Compile
    chatbot = graph.compile()
    return chatbot

# -------------------
# 6. Test Function
# -------------------
async def main():
    """
    Test the RAG chatbot with multiple scenarios
    """
    chatbot = build_graph()
    
    # Test 1: Document question (should use search_document tool)
    print("\n" + "="*60)
    print("Test 1: Document Question")
    print("="*60)
    result1 = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content='Who is the author of the document?')]
        },
        config={"configurable": {"thread_id": "test1"}}
    )
    print(f"Answer: {result1['messages'][-1].content}\n")
    
    # Test 2: Math question (should use calculator tool)
    print("="*60)
    print("Test 2: Math Question")
    print("="*60)
    result2 = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content='What is 456 multiplied by 789?')]
        },
        config={"configurable": {"thread_id": "test2"}}
    )
    print(f"Answer: {result2['messages'][-1].content}\n")
    
    # Test 3: Story content question (should use search_document tool)
    print("="*60)
    print("Test 3: Story Content Question")
    print("="*60)
    result3 = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content='What is the story about? Give me a brief summary.')]
        },
        config={"configurable": {"thread_id": "test3"}}
    )
    print(f"Answer: {result3['messages'][-1].content}\n")
    
    # Test 4: General question (should NOT use any tool)
    print("="*60)
    print("Test 4: General Question")
    print("="*60)
    result4 = await chatbot.ainvoke(
        {
            'messages': [HumanMessage(content='What is the capital of France?')]
        },
        config={"configurable": {"thread_id": "test4"}}
    )
    print(f"Answer: {result4['messages'][-1].content}\n")


if __name__ == '__main__':
    asyncio.run(main())
