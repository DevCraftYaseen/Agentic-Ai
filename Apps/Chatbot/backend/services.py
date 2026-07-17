from graph import chat_bot, conn, llm
from langchain_core.messages import HumanMessage

def get_or_create_title(thread_id: str, first_message: str) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    
    # If title already exists, return it
    if row:
        return row[0]
    
    # If this is a new thread, generate a dynamic title
    prompt = f"Generate a brief, 3 to 4 word title for a chat that starts with this message: '{first_message}'. Respond ONLY with the title, no quotes or punctuation."
    title = llm.invoke(prompt).content.strip()
    
    # Save the new title
    cursor.execute("INSERT INTO chat_titles (thread_id, title) VALUES (?, ?)", (thread_id, title))
    conn.commit()
    
    return title

def get_all_threads():
    cursor = conn.cursor()
    # Order by most recently created/updated (SQLite ROWID)
    cursor.execute("SELECT thread_id, title FROM chat_titles ORDER BY ROWID DESC")
    rows = cursor.fetchall()
    return [{"thread_id": row[0], "title": row[1]} for row in rows]

def get_thread_history(thread_id: str):
    state = chat_bot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    
    formatted_messages = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        formatted_messages.append({"role": role, "content": msg.content})
        
    return formatted_messages