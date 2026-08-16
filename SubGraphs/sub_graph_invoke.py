"""
SubGraph Invoke Pattern with Separate States
----------------------------------------------
This demonstrates how to use a subgraph with its own separate state structure.
The parent graph invokes the subgraph and maps data between different state types.

Use Case: Translation pipeline with isolated state management.
Advantage: Subgraph can be reused in different contexts with different parent states.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# ==================== STATE DEFINITIONS ====================

class TranslationState(TypedDict):
    """State structure specific to the translation subgraph"""
    source_text: str
    translated_text: str


class QuestionAnswerState(TypedDict):
    """State structure for the parent Q&A workflow"""
    question: str
    english_answer: str
    urdu_answer: str


# Initialize LLM
llm = ChatOllama(model='llama3.1:8b')


# ==================== TRANSLATION SUB GRAPH ====================

def translate_to_urdu(state: TranslationState) -> dict:
    """
    Translate English text to Urdu with high fidelity.
    
    This is a reusable translation node that only knows about TranslationState.
    It can be used in any graph that needs English-to-Urdu translation.
    
    Args:
        state: Contains 'source_text' with English text to translate
        
    Returns:
        Dictionary with 'translated_text' containing Urdu translation
    """
    english_text = state.get('source_text', '').strip()
    
    # Input validation
    if not english_text:
        return {'translated_text': 'خالی متن (Empty text)'}
    
    # Professional translation prompt with clear instructions
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert English-to-Urdu translator with deep understanding of both languages.

Translation Guidelines:
1. Translate accurately while preserving the original meaning and nuance
2. Maintain the formality level and tone of the source text
3. Keep technical terms in English if they're commonly used in Urdu (e.g., "AI", "Python", "internet")
4. Use proper Urdu grammar and natural phrasing
5. Do NOT add explanations, notes, or commentary
6. Do NOT include English text in the output
7. Output ONLY the Urdu translation in proper Urdu script

Quality standard: Your translation should read as if it was originally written in Urdu."""),
        ("user", "Translate the following English text to Urdu:\n\n{text}")
    ])
    
    try:
        chain = translation_prompt | llm
        result = chain.invoke({"text": english_text})
        return {'translated_text': result.content.strip()}
    except Exception as e:
        return {'translated_text': f'ترجمہ میں خرابی: {str(e)}'}


# Build translation subgraph with its own state
translation_builder = StateGraph(TranslationState)
translation_builder.add_node('translate_to_urdu', translate_to_urdu)
translation_builder.add_edge(START, "translate_to_urdu")
translation_builder.add_edge("translate_to_urdu", END)

# Compile the reusable translation subgraph
translation_engine = translation_builder.compile()


# ==================== PARENT Q&A GRAPH ====================

def generate_english_answer(state: QuestionAnswerState) -> dict:
    """
    Generate a clear, informative answer to the user's question in English.
    
    Args:
        state: Contains 'question' with the user's query
        
    Returns:
        Dictionary with 'english_answer' containing the response
    """
    question = state.get('question', '').strip()
    
    # Input validation
    if not question:
        return {'english_answer': 'Please provide a question.'}
    
    # Structured prompt for high-quality answers
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable and helpful AI assistant specializing in clear explanations.

Response Guidelines:
1. Provide accurate, well-structured answers
2. Use simple, accessible language suitable for a general audience
3. Be concise but comprehensive - answer thoroughly without unnecessary verbosity
4. For complex topics, start with a simple explanation, then add layers of detail
5. Use examples or analogies when they aid understanding
6. Aim for 2-5 sentences for simple questions, more for complex ones
7. If a question is ambiguous, answer the most reasonable interpretation

Tone: Professional yet approachable, informative and friendly."""),
        ("user", "{question}")
    ])
    
    try:
        chain = answer_prompt | llm
        result = chain.invoke({"question": question})
        return {'english_answer': result.content.strip()}
    except Exception as e:
        return {'english_answer': f'Error: Unable to generate answer - {str(e)}'}


def invoke_translation(state: QuestionAnswerState) -> dict:
    """
    Wrapper function that invokes the translation subgraph.
    
    This demonstrates state mapping between parent and subgraph:
    - Maps parent's 'english_answer' → subgraph's 'source_text'
    - Maps subgraph's 'translated_text' → parent's 'urdu_answer'
    
    Args:
        state: Parent state with 'english_answer' to translate
        
    Returns:
        Dictionary with 'urdu_answer' containing translation
    """
    english_text = state.get('english_answer', '').strip()
    
    if not english_text:
        return {'urdu_answer': 'کوئی جواب نہیں (No answer to translate)'}
    
    try:
        # Invoke the translation subgraph with mapped state
        translation_result = translation_engine.invoke({
            'source_text': english_text
        })
        
        # Map the subgraph output back to parent state
        return {'urdu_answer': translation_result['translated_text']}
    
    except Exception as e:
        return {'urdu_answer': f'Translation error: {str(e)}'}


# Build parent Q&A graph
qa_builder = StateGraph(QuestionAnswerState)

# Add nodes
qa_builder.add_node('generate_english_answer', generate_english_answer)
qa_builder.add_node('invoke_translation', invoke_translation)

# Add edges - sequential workflow
qa_builder.add_edge(START, "generate_english_answer")
qa_builder.add_edge("generate_english_answer", "invoke_translation")
qa_builder.add_edge("invoke_translation", END)

# Compile the parent graph
bilingual_qa_system = qa_builder.compile()


# ==================== MAIN EXECUTION ====================

def main():
    """
    Run interactive demo of the bilingual Q&A system.
    Demonstrates state mapping between parent and subgraph.
    """
    
    print("=" * 75)
    print("🌐 Bilingual Q&A System (Separate State Pattern)")
    print("=" * 75)
    print("\nFeatures:")
    print("  • Generates answers in English")
    print("  • Translates to Urdu using isolated subgraph")
    print("  • Demonstrates state mapping between graphs")
    print("\n" + "=" * 75 + "\n")
    
    # Example questions showcasing different complexity levels
    test_questions = [
        "What is AI in simple words?",
        "How does machine learning work?",
        "Explain the difference between Python and JavaScript.",
        "What is the capital of Pakistan?"
    ]
    
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 75}")
        print(f"Question {idx}: {question}")
        print('─' * 75)
        
        try:
            # Invoke the bilingual Q&A system
            result = bilingual_qa_system.invoke({'question': question})
            
            print(f"\n📝 English Answer:")
            print(f"   {result.get('english_answer', 'N/A')}")
            
            print(f"\n🌍 Urdu Translation (اردو ترجمہ):")
            print(f"   {result.get('urdu_answer', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ Error processing question: {str(e)}")
        
        print()
    
    print("=" * 75)
    print("✅ Demo completed successfully!")
    print("=" * 75)
    
    # Show the advantage of separate states
    print("\n💡 Key Advantage of This Pattern:")
    print("   The translation subgraph can be reused in ANY context")
    print("   that needs English-to-Urdu translation, not just Q&A!")
    print()


if __name__ == "__main__":
    main()