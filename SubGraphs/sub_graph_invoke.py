"""
SubGraph Invoke Pattern with Separate States and Persistence
-------------------------------------------------------------
Use a subgraph with its own separate state structure.
The parent graph invokes the subgraph and maps data between different state types.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import time

load_dotenv()


class TranslationState(TypedDict):
    source_text: str
    translated_text: str


class QuestionAnswerState(TypedDict):
    question: str
    english_answer: str
    urdu_answer: str


llm = ChatOllama(model='llama3.1:8b')


# ==================== TRANSLATION SUB GRAPH ====================

def translate_to_urdu(state: TranslationState) -> dict:
    english_text = state.get('source_text', '').strip()
    
    if not english_text:
        return {'translated_text': 'خالی متن (Empty text)'}
    
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


translation_builder = StateGraph(TranslationState)
translation_builder.add_node('translate_to_urdu', translate_to_urdu)
translation_builder.add_edge(START, "translate_to_urdu")
translation_builder.add_edge("translate_to_urdu", END)

translation_engine = translation_builder.compile()


# ==================== PARENT Q&A GRAPH ====================

def generate_english_answer(state: QuestionAnswerState) -> dict:
    question = state.get('question', '').strip()
    
    if not question:
        return {'english_answer': 'Please provide a question.'}
    
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
    english_text = state.get('english_answer', '').strip()
    
    if not english_text:
        return {'urdu_answer': 'کوئی جواب نہیں (No answer to translate)'}
    
    try:
        translation_result = translation_engine.invoke({
            'source_text': english_text
        })
        
        return {'urdu_answer': translation_result['translated_text']}
    
    except Exception as e:
        return {'urdu_answer': f'Translation error: {str(e)}'}


qa_builder = StateGraph(QuestionAnswerState)
qa_builder.add_node('generate_english_answer', generate_english_answer)
qa_builder.add_node('invoke_translation', invoke_translation)
qa_builder.add_edge(START, "generate_english_answer")
qa_builder.add_edge("generate_english_answer", "invoke_translation")
qa_builder.add_edge("invoke_translation", END)

conn = sqlite3.connect(database="bilingual_qa_invoke.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

bilingual_qa_system = qa_builder.compile(checkpointer=checkpointer)


# ==================== MAIN ====================

def main():
    thread_id = f"session_{int(time.time())}"
    config = {'configurable': {'thread_id': thread_id}}
    
    print("=" * 75)
    print("💬 Bilingual Q&A System (Separate State Pattern)")
    print("=" * 75)
    print(f"\nSession ID: {thread_id}")
    print("Type your questions in English. Type 'exit' to quit.\n")
    print("=" * 75 + "\n")
    
    try:
        while True:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'bye']:
                break
            
            try:
                result = bilingual_qa_system.invoke(
                    {'question': question},
                    config=config
                )
                
                print(f"\n📝 English: {result.get('english_answer', 'N/A')}")
                print(f"🌍 Urdu: {result.get('urdu_answer', 'N/A')}\n")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    print("\n" + "=" * 75)
    print(f"� Session saved with ID: {thread_id}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
