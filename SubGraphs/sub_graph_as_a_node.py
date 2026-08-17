"""
SubGraph as a Node Pattern with Persistence
--------------------------------------------
Embed a compiled subgraph directly as a node in a parent graph.
The subgraph shares the same state structure as the parent graph.
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


class ParentState(TypedDict):
    question: str
    english_answer: str
    urdu_answer: str


llm = ChatOllama(model='llama3.1:8b')


# ==================== SUB GRAPH ====================

def translate_answer(state: ParentState) -> dict:
    english_text = state.get('english_answer', '').strip()
    
    if not english_text:
        return {'urdu_answer': 'خالی جواب (Empty answer)'}
    
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


subgraph_builder = StateGraph(ParentState)
subgraph_builder.add_node('translate_answer', translate_answer)
subgraph_builder.add_edge(START, "translate_answer")
subgraph_builder.add_edge("translate_answer", END)

translation_subgraph = subgraph_builder.compile()


# ==================== PARENT GRAPH ====================

def generate_answer(state: ParentState) -> dict:
    question = state.get('question', '').strip()
    
    if not question:
        return {'english_answer': 'No question provided.'}
    
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


parent_builder = StateGraph(ParentState)
parent_builder.add_node('generate_answer', generate_answer)
parent_builder.add_node('translate', translation_subgraph)
parent_builder.add_edge(START, "generate_answer")
parent_builder.add_edge("generate_answer", "translate")
parent_builder.add_edge("translate", END)

conn = sqlite3.connect(database="bilingual_qa.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

bilingual_qa_agent = parent_builder.compile(checkpointer=checkpointer)


# ==================== MAIN ====================

def main():
    thread_id = f"session_{int(time.time())}"
    config = {'configurable': {'thread_id': thread_id}}
    
    print("=" * 70)
    print("💬 Bilingual Q&A Agent (English → Urdu)")
    print("=" * 70)
    print(f"\nSession ID: {thread_id}")
    print("Type your questions in English. Type 'exit' to quit.\n")
    print("=" * 70 + "\n")
    
    try:
        while True:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'bye']:
                break
            
            try:
                result = bilingual_qa_agent.invoke(
                    {'question': question},
                    config=config
                )
                
                print(f"\n📝 English: {result['english_answer']}")
                print(f"🌍 Urdu: {result['urdu_answer']}\n")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    print("\n" + "=" * 70)
    print(f"💾 Session saved with ID: {thread_id}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()