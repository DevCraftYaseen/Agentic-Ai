# 🤖 Advanced AI Chatbot

A full-stack AI chatbot application with multiple advanced capabilities including Tools, RAG, HITL, and more.

## ✨ Features

### 🔧 **Tools Integration**
- **Web Search**: Search the web for current information using DuckDuckGo
- **Calculator**: Perform mathematical calculations
- **Stock Prices**: Get real-time stock market data
- **Document Q&A**: Query uploaded PDF documents using RAG

### 📚 **RAG (Retrieval-Augmented Generation)**
- Upload PDF documents
- Vector store with FAISS
- Semantic search across documents
- Context-aware responses

### 🔒 **HITL (Human-in-the-Loop)**
- Stock purchase approvals
- Interactive approval modals
- Safe execution of sensitive actions

### 💬 **Chat Features**
- Real-time streaming responses
- Conversation history with smart titles
- Thread management (create, load, delete)
- Markdown rendering
- Mobile responsive design

### 🎨 **Design**
- Minimalist black and white theme
- Clean, modern interface
- Smooth animations and transitions
- Fully responsive

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Gemini API Key
- Alpha Vantage API Key (optional, for stock prices)

### Backend Setup

1. Navigate to backend directory:
```bash
cd Apps/Chatbot/backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
STOCK_API_KEY=your_alpha_vantage_key_here  # Optional
```

5. Run the backend:
```bash
python main_advanced.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd Apps/Chatbot/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## 📖 Usage Guide

### Basic Chat
1. Click "New Chat" to start a conversation
2. Type your message in the input box
3. Press Enter or click Send

### Using Tools
The AI will automatically use tools when appropriate:

- **Web Search**: "What's the latest news about AI?"
- **Calculator**: "Calculate 234 * 567"
- **Stock Prices**: "What's the current price of AAPL?"
- **Document Q&A**: "What does the document say about...?"

### Uploading Documents (RAG)
1. Click "Upload Document" in the sidebar
2. Select a PDF file
3. Wait for processing
4. Ask questions about the document content

### Stock Purchase (HITL Demo)
1. Ask: "Buy 10 shares of TSLA"
2. Approval modal will appear
3. Approve or decline the purchase
4. See the result in the chat

### Managing Conversations
- **New Chat**: Start a fresh conversation
- **Load Chat**: Click any chat in the sidebar
- **Delete Chat**: Hover over a chat and click the trash icon

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: REST API and SSE streaming
- **LangGraph**: Agent orchestration
- **LangChain**: LLM framework
- **Google Gemini**: Language model
- **FAISS**: Vector store for RAG
- **SQLite**: Persistence and checkpointing

### Frontend Stack
- **Next.js 15**: React framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Lucide React**: Icons
- **React Markdown**: Message rendering

### Key Components

#### Backend
- `advanced_graph.py`: Main LangGraph with tools, RAG, and HITL
- `main_advanced.py`: FastAPI server with all endpoints
- `requirements.txt`: Python dependencies

#### Frontend
- `AdvancedChatClient.tsx`: Main chat logic
- `Sidebar.tsx`: Thread management
- `AdvancedChatArea.tsx`: Message display
- `ApprovalModal.tsx`: HITL approval UI
- `DocumentUpload.tsx`: RAG document management

## 📡 API Endpoints

### Chat
- `POST /api/chat/stream` - Stream chat responses
- `POST /api/chat/approval` - Handle HITL approvals
- `GET /api/interrupts/{thread_id}` - Check pending approvals

### Threads
- `GET /api/threads` - Get all chat threads
- `GET /api/threads/{thread_id}` - Get thread history
- `DELETE /api/threads/{thread_id}` - Delete a thread

### Documents (RAG)
- `POST /api/documents/upload` - Upload PDF
- `DELETE /api/documents/clear` - Clear all documents

### Health
- `GET /` - API info
- `GET /api/health` - Health check

## 🎯 Example Prompts

### Web Search
```
What are the latest developments in AI?
Search for recent news about electric vehicles
```

### Calculator
```
Calculate the area of a circle with radius 5
What is 2^10?
```

### Stock Prices
```
What's the current price of Tesla stock?
Show me AAPL stock info
```

### Document Q&A
```
Summarize the main points from the document
What does the document say about chapter 3?
```

### HITL (Stock Purchase)
```
I want to buy 50 shares of Google
Purchase 100 shares of MSFT
```

## 🔧 Configuration

### Environment Variables
```env
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
STOCK_API_KEY=your_alpha_vantage_key
```

### Model Configuration
Edit `advanced_graph.py`:
```python
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    streaming=True,
    temperature=0.7
)
```

## 🐛 Troubleshooting

### Backend Issues
- **Import errors**: Make sure virtual environment is activated
- **API key errors**: Check `.env` file exists and has valid keys
- **Port 8000 in use**: Change port in `main_advanced.py`

### Frontend Issues
- **Connection refused**: Ensure backend is running on port 8000
- **Styling issues**: Run `npm install` to ensure all dependencies are installed
- **Build errors**: Delete `node_modules` and `.next`, then reinstall

### RAG Issues
- **Upload fails**: Check file is PDF and size is reasonable
- **No results**: Ensure documents are uploaded before querying
- **Vector store error**: Delete `vector_store` directory and re-upload

## 📝 Notes

- Stock purchases are simulated for demonstration purposes
- HITL approvals demonstrate the pattern but don't execute real trades
- Web search uses DuckDuckGo (no API key required)
- Document processing may take time for large PDFs

## 🚧 Future Enhancements

- [ ] MCP (Model Context Protocol) integration
- [ ] More tool integrations
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Export conversations
- [ ] Custom tool creation UI
- [ ] Advanced RAG with multiple vector stores
- [ ] User authentication

## 👨‍💻 Built By

**Yaseen Khan (DevCraftYaseen)**
- Full Stack Web Developer
- AI Automation Engineer

## 📄 License

This project is for educational and portfolio purposes.

---

**Day 40: Advanced Full-Stack AI Chatbot** 🎉
