"""
SubGraph as a Node Pattern
----------------------------
This demonstrates how to embed a compiled subgraph directly as a node in a parent graph.
The subgraph shares the same state structure as the parent graph.

Use Case: Translation pipeline that generates English answers and translates to Urdu.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class ParentState(TypedDict):
    """Shared state structure for both parent and subgraph"""
    question: str
    english_answer: str
    urdu_answer: str

# Initialize LLM
llm = ChatOllama(model='llama3.1:8b')


# ==================== SUB GRAPH ====================

def translate_answer(state: ParentState) -> dict:
    """
    Translate English answer to Urdu with high accuracy.
    
    Args:
        state: Contains 'english_answer' key with text to translate
        
    Returns:
        Dictionary with 'urdu_answer' key containing translation
    """
    english_text = state.get('english_answer', '').strip()
    
    # Validation
    if not english_text:
        return {'urdu_answer': 'خالی جواب (Empty answer)'}
    
    # translation prompt
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional English-to-Urdu translator.

Rules:
1. Translate the text accurately while preserving the original meaning
2. Maintain the tone and style of the original text
3. Do NOT add explanations, notes, or extra content
4. Do NOT translate technical terms that are commonly used in Urdu (e.g., "AI", "computer")
5. Return ONLY the Urdu translation, nothing else

Output the translation in proper Urdu script."""),
        ("user", "Translate this English text to Urdu:\n\n{text}")
    ])
    
    try:
        chain = translation_prompt | llm
        result = chain.invoke({"text": english_text})
        return {'urdu_answer': result.content.strip()}
    except Exception as e:
        return {'urdu_answer': f'Translation error: {str(e)}'}


# Build subgraph
subgraph_builder = StateGraph(ParentState)
subgraph_builder.add_node('translate_answer', translate_answer)
subgraph_builder.add_edge(START, "translate_answer")
subgraph_builder.add_edge("translate_answer", END)

# Compile the subgraph
translation_subgraph = subgraph_builder.compile()


# ==================== PARENT GRAPH ====================

def generate_answer(state: ParentState) -> dict:
    """
    Generate a clear, concise answer to the user's question in English.
    
    Args:
        state: Contains 'question' key with user query
        
    Returns:
        Dictionary with 'english_answer' key containing response
    """
    question = state.get('question', '').strip()
    
    # Validation
    if not question:
        return {'english_answer': 'No question provided.'}
    
    # Optimized answer generation prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and knowledgeable AI assistant.

Your guidelines:
1. Provide clear, accurate, and concise answers
2. Use simple language that anyone can understand
3. Be direct - avoid unnecessary elaboration
4. If the question is unclear, answer based on the most likely interpretation
5. For technical topics, explain in simple terms first, then add details if needed
6. Keep answers between 2-4 sentences unless more detail is genuinely required

Tone: Friendly, professional, and informative."""),
        ("user", "{question}")
    ])
    
    try:
        chain = answer_prompt | llm
        result = chain.invoke({"question": question})
        return {'english_answer': result.content.strip()}
    except Exception as e:
        return {'english_answer': f'Error generating answer: {str(e)}'}


# Build parent graph
parent_builder = StateGraph(ParentState)

# Add nodes (subgraph is added as a regular node)
parent_builder.add_node('generate_answer', generate_answer)
parent_builder.add_node('translate', translation_subgraph)

# Add edges - linear flow
parent_builder.add_edge(START, "generate_answer")
parent_builder.add_edge("generate_answer", "translate")
parent_builder.add_edge("translate", END)

# Compile the parent graph
bilingual_qa_agent = parent_builder.compile()


# ==================== MAIN EXECUTION ====================

def main():
    """Run the bilingual Q&A agent with example questions."""
    
    print("=" * 70)
    print("🌐 Bilingual Q&A Agent (English → Urdu)")
    print("=" * 70)
    print("\nThis agent answers questions in English and translates to Urdu.\n")
    
    # Test questions
    test_questions = [
        "What is AI in simple words?",
        "How does machine learning work?",
        "What is the capital of Pakistan?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print('='*70)
        
        try:
            result = bilingual_qa_agent.invoke({'question': question})
            
            print(f"\n📝 English Answer:")
            print(f"   {result['english_answer']}")
            
            print(f"\n🌍 Urdu Translation:")
            print(f"   {result['urdu_answer']}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ Demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()