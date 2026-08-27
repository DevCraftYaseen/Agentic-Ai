# Agentic AI: My Engineering Journey 🤖

Documenting my practical journey as an AI Automation and Agents Engineer. This repository serves as my daily practice ground for coding, testing, and evolving autonomous AI workflows. It tracks my progression from core LangChain concepts to building stateful LangGraph agents with advanced memory systems, culminating in a production-ready full-stack application.

## 👨‍💻 About the Developer
Built by **Yaseen Khan (DevCraftYaseen)**  
Full Stack Web Developer & AI Automation Engineer
* **Tech Stack:** Python, LangChain, LangGraph, FastAPI, Next.js (App Router), React, PostgreSQL, SQLite, FAISS, and Ollama.

---

## 🗺️ The Learning Roadmap

### Phase 1: LangChain Foundations (Days 1-12)
**Objective:** Master core LangChain components for building LLM applications

* **Day 1:** AI Engineering setup, dependencies, and core LangChain models
* **Day 2:** Mastered LangChain prompt templates and UI integrations
* **Day 3 & 4:** Implemented Structured Outputs and Output Parsers
* **Day 6:** Built execution Chains to link LLMs and prompts
* **Day 7:** Configured Document Loaders for data ingestion
* **Day 8:** Applied Text Splitters for efficient data chunking
* **Day 9:** Set up Vector Stores for embedding storage
* **Day 10:** Implemented Retrievers for context fetching
* **Day 11:** Created custom Tools and orchestrated a ReAct Agent
* **Day 12:** Completed LangChain Projects - Chat Doc and YouChat

### Phase 2: LangGraph & Stateful Agents (Days 13-20)
**Objective:** Build stateful, multi-step reasoning agents using LangGraph

* **Day 13:** Started LangGraph with a basic BMI calculation state graph
* **Day 14:** Integrated basic LLM node execution within LangGraph
* **Day 15:** Implemented sequential prompt chaining in graph nodes
* **Day 16:** Built BAT result evaluation and routing logic
* **Day 17:** Created an automated essay evaluation agent
* **Day 18:** Engineered a mathematical reasoning agent for quadratic equations
* **Day 19:** Developed a sentiment analysis node for text classification
* **Day 20:** Built an automated X (Twitter) post generation and formatting agent

### Phase 3: Persistence & Full-Stack Integration (Days 21-26)
**Objective:** Add state persistence and build production-ready full-stack architecture

* **Day 21:** Built a basic conversational chatbot using LangGraph
* **Day 22:** Added state persistence and memory to LangGraph agents
* **Day 23:** Developed a chatbot with a Streamlit UI and LangGraph backend
* **Day 24:** Implemented UI streaming and threading for multiple active sidebar chats
* **Day 25:** Integrated SQLite database for persistent chatbot memory and state tracking
* **Day 26 (Backend):** Migrated LangGraph to FastAPI server with SSE, enabled Gemini token streaming, and resolved AIMessageChunk filtering bug
* **Day 26 (Frontend):** Built Next.js UI with ChatArea/Sidebar, integrated react-markdown, fixed React Strict Mode state mutations, and added UI/UX polish

### Phase 4: Observability & Tool Integration (Days 27-35)
**Objective:** Add observability, external tools, and MCP integration for production monitoring

* **Day 27:** Introduced LangSmith for application observability, successfully tracing simple LLM calls and sequential chains
* **Day 28:** Implemented and evaluated multiple iterations of Retrieval-Augmented Generation (RAG) architectures (v1-v4) using LangSmith tracing
* **Day 29:** Traced ReAct agent executions and stateful LangGraph workflows using LangSmith to monitor autonomous decision-making
* **Day 30:** Integrated LangSmith into the chatbot architectures, utilizing `thread_id` metadata to track and debug specific multi-turn conversational sessions
* **Day 31:** Upgraded the LangGraph chatbot into a tool-calling ReAct agent by integrating external APIs (Search, Calculator, and real-time Stock Prices)
* **Day 32:** Expanded the agent's toolset with real-time currency exchange and conversion tools, and optimized the frontend for highly efficient token streaming
* **Day 33:** Learned the Model Context Protocol (MCP) architecture in detail for modular tool integration
* **Day 34:** Built an MCP Client using async code
* **Day 34.2:** Added the math MCP server to the mcp_chatbot
* **Day 34.3:** Added expense tracker MCP tool to the mcp chatbot
* **Day 35:** Converted MCP chatbot to production-ready architecture with dedicated async event loop, AsyncSqliteSaver checkpointer, and Streamlit frontend; configured general-purpose AI assistant with intelligent tool routing for math operations and expense tracking

### Phase 5: RAG & Advanced Retrieval (Days 36-38)
**Objective:** Build intelligent document Q&A systems with vector search and retrieval

* **Day 36:** Built RAG-as-a-tool system with FAISS vector store, document retrieval, and LangGraph orchestration for intelligent context-based responses
* **Day 37:** Implemented RAG chatbot with dynamic PDF management, vector search, and Streamlit frontend for intelligent document Q&A
* **Day 37.2:** Fixed bugs and issues related to streaming and file management in the RAG Chatbot
* **Day 38:** Added Search tool, improved UI and streaming responses as well as tool outputs in the RAG Chatbot

### Phase 6: Human-in-the-Loop & Subgraphs (Days 39-44)
**Objective:** Implement approval workflows and modular subgraph patterns

* **Day 39:** Implemented Human-in-the-Loop (HITL) pattern with LangGraph interrupts; built basic approval chatbot and stock trading agent with real-time purchase authorization
* **Day 40:** Implemented LangGraph subgraph patterns (embedded & invoked) with optimized prompt engineering; built bilingual Q&A system with English-to-Urdu translation pipeline and production-grade error handling
* **Day 41:** Added persistence using SQLite to bilingual Q&A system with English-to-Urdu translation
* **Day 42:** Improved backend graph with better prompts and error handling
* **Day 43:** Built frontend using Next.js and Tailwind CSS for the chatbot
* **Day 44:** Fixed streaming pipeline end-to-end and resolved HITL approval detection bug

### Phase 7: Memory Systems & Agent Intelligence (Days 45-49)
**Objective:** Implement Short-Term and Long-Term Memory for personalized, context-aware agents

* **Day 45:** Learned deeply about Memory in LLMs: Short-Term Memory (STM) and Long-Term Memory (LTM)
* **Day 46:** Implemented Short-Term Memory (STM) in LangGraph with async PostgreSQL checkpointer, real-time streaming, and message trimming for context window management
* **Day 47:** Added summarization and message deletion to the STM for efficient context management
* **Day 48:** Implemented Long-Term Memory (LTM) in LangGraph with InMemoryStore, namespace-based memory isolation, semantic search using embeddings, and personalized context retrieval
* **Day 49:** Advanced LTM implementation with PostgreSQL persistence, memory deduplication to prevent redundant storage, controlled memory writing with structured extraction, and terminal-based chatbot with optimized prompts for natural personalized responses

---

## ⚙️ Local Setup & Installation

This project utilizes a decoupled full-stack architecture. You will need to run the backend and frontend simultaneously in separate terminal windows.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/agentic-ai.git
cd agentic-ai
```

### 2. Start the FastAPI Backend
```bash
# Navigate to the backend directory
cd Apps/Chatbot/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables (create a .env file)
# DB_URI=postgresql://user:password@localhost:5432/dbname
# STOCK_API_KEY=your_key_here

# Run the FastAPI server
uvicorn main:app --reload
```

### 3. Start the Next.js Frontend
```bash
# Open a new terminal and navigate to the frontend directory
cd Apps/Chatbot/frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

### 4. Memory Systems Setup
```bash
# For PostgreSQL-based memory (STM & Advanced LTM)
# Ensure PostgreSQL is running and DB_URI is configured in .env

# Run STM chatbot with async streaming
python Memory/stm.py

# Run Advanced LTM chatbot with deduplication
python Memory/LTM/advanced_ltm.py
```

---

## 🛡️ License
This project is for educational and portfolio purposes. Feel free to explore the code!

---

## 📊 Project Structure
```
Agentic-Ai/
├── Apps/
│   └── Chatbot/
│       ├── backend/          # FastAPI + LangGraph
│       └── frontend/         # Next.js + React
├── LangChain/               # Phase 1: LangChain basics
├── LangGraph/               # Phase 2: Stateful agents
├── Memory/
│   ├── STM/                 # Short-Term Memory
│   └── LTM/                 # Long-Term Memory
├── HITL/                    # Human-in-the-Loop patterns
└── SubGraphs/               # Modular graph patterns
```
