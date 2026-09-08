"""
Test script to verify CodeCraft chatbot setup
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("CodeCraft Chatbot Setup Verification")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
print("-" * 60)

required_vars = {
    "CODECRAFT_API_KEY": "CodeCraft API",
    "CODECRAFT_BASE_URL": "CodeCraft Base URL",
    "CODECRAFT_MODEL": "CodeCraft Model",
    "TAVILY_API_KEY": "Tavily Search API"
}

all_ok = True
for var, name in required_vars.items():
    value = os.getenv(var)
    if value and value != "cc_your_key_here":
        print(f"✓ {name}: {'*' * 20}{value[-8:]}")
    else:
        print(f"✗ {name}: NOT SET")
        all_ok = False

# Test CodeCraft API
print("\n2. Testing CodeCraft API:")
print("-" * 60)
try:
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("CODECRAFT_API_KEY"),
        base_url=os.getenv("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")
    )
    
    response = client.chat.completions.create(
        model=os.getenv("CODECRAFT_MODEL", "gpt-5.6-sol"),
        messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
        max_tokens=10
    )
    
    print(f"✓ API Response: {response.choices[0].message.content}")
    print(f"✓ Model: {response.model}")
except Exception as e:
    print(f"✗ API Error: {str(e)}")
    all_ok = False

# Test Tavily
print("\n3. Testing Tavily Search:")
print("-" * 60)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    tool = TavilySearchResults(max_results=1)
    results = tool.invoke({"query": "Python programming"})
    
    if results:
        print(f"✓ Search working: Found {len(results)} result(s)")
    else:
        print("✗ No results returned")
except Exception as e:
    print(f"✗ Search Error: {str(e)}")
    all_ok = False

# Test SQLite & Memory Store
print("\n4. Testing SQLite & Memory Store:")
print("-" * 60)
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.memory import InMemoryStore
    import asyncio
    
    # Test checkpointer
    async def test_checkpointer():
        checkpointer = AsyncSqliteSaver.from_conn_string("test_checkpoints.db")
        return checkpointer
    
    loop = asyncio.new_event_loop()
    checkpointer = loop.run_until_complete(test_checkpointer())
    print("✓ AsyncSqliteSaver initialized")
    
    # Test store
    store = InMemoryStore()
    print("✓ InMemoryStore initialized")
    
    # Clean up test DB
    import os
    if os.path.exists("test_checkpoints.db"):
        os.remove("test_checkpoints.db")
    
except Exception as e:
    print(f"✗ Database Error: {str(e)}")
    all_ok = False

# Summary
print("\n" + "=" * 60)
if all_ok:
    print("✓ All checks passed! You're ready to run the chatbot.")
    print("\nNext steps:")
    print("1. Run backend:  python main.py")
    print("2. Run frontend: streamlit run ../frontend/app.py")
else:
    print("✗ Some checks failed. Please fix the issues above.")
print("=" * 60)
