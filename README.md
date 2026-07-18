# Agentic AI: My Engineering Journey 🤖

Documenting my practical journey as an AI Automation and Agents Engineer. This repository serves as my daily practice ground for coding, testing, and evolving autonomous AI workflows. It tracks my progression from core LangChain concepts to building stateful LangGraph agents, ultimately culminating in a decoupled, real-time full-stack application.

## 👨‍💻 About the Developer
Built by **Yaseen Khan (DevCraftYaseen)** 
Full Stack Web Developer & AI Automation Engineer
* **Tech Stack:** Python, LangChain, LangGraph, FastAPI, Next.js (App Router), React, SQLite, and Chroma.

---

## 🗺️ The Learning Roadmap

### Phase 1: LangChain Foundations
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

### Phase 2: LangGraph & Stateful Agents
* **Day 13:** Started LangGraph with a basic BMI calculation state graph
* **Day 14:** Integrated basic LLM node execution within LangGraph
* **Day 15:** Implemented sequential prompt chaining in graph nodes
* **Day 16:** Built BAT result evaluation and routing logic
* **Day 17:** Created an automated essay evaluation agent
* **Day 18:** Engineered a mathematical reasoning agent for quadratic equations
* **Day 19:** Developed a sentiment analysis node for text classification
* **Day 20:** Built an automated X (Twitter) post generation and formatting agent

### Phase 3: Full-Stack AI Integration
* **Day 21:** Built a basic conversational chatbot using LangGraph
* **Day 22:** Added state persistence and memory to LangGraph agents
* **Day 23:** Developed a chatbot with a Streamlit UI and LangGraph backend
* **Day 24:** Implemented UI streaming and threading for multiple active sidebar chats
* **Day 25:** Integrated SQLite database for persistent chatbot memory and state tracking
* **Day 26:** Backend: Migrated LangGraph to FastAPI server with SSE, enabled Gemini token streaming, and resolved AIMessageChunk filtering bug
* **Day 26:** Frontend: Built Next.js UI with ChatArea/Sidebar, integrated react-markdown, fixed React Strict Mode state mutations, and added UI/UX polish

### Phase 4: Observability & Evaluation
* **Day 27:** Introduced LangSmith for application observability, successfully tracing simple LLM calls and sequential chains.

---

## ⚙️ Local Setup & Installation

This project utilizes a decoupled full-stack architecture. You will need to run the backend and frontend simultaneously in separate terminal windows.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/agentic-ai.git](https://github.com/yourusername/agentic-ai.git)
cd agentic-ai

2. Start the FastAPI Backend
```bash
# Navigate to the backend directory
cd Apps/Chatbot/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables (create a .env file)
# OPENAI_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here

# Run the FastAPI server
uvicorn main:app --reload

3. Start the Next.js Frontend
```bash
# Open a new terminal and navigate to the frontend directory
cd Apps/Chatbot/frontend

# Install dependencies
npm install

# Run the development server
npm run dev

🛡️ License
This project is for educational and portfolio purposes. Feel free to explore the code!

---

### **Step 4: Stage and Push the Restructured App**
Now that the nested Git folders are gone and the ignore rules are unified at the root, you can stage the files exactly how you wanted:

```bash
# Stage the consolidated gitignore and updated README
git add .gitignore
git add README.md

# Stage the frontend and backend as standard directories
git add Apps/Chatbot/backend/
git add Apps/Chatbot/frontend/

# Commit the massive full-stack migration
git commit -m "Day 26: Migrated to a full-stack architecture with Next.js frontend, FastAPI backend, SSE token streaming, and a unified root git configuration"

# Push to GitHub
git push