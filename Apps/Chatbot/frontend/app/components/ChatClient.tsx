// components/ChatClient.tsx
"use client";

import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import ChatArea from "./ChatArea";

export type Message = { role: "user" | "assistant"; content: string };
export type Thread = { thread_id: string; title: string };

export default function ChatClient() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    fetchThreads();
    startNewChat();
  }, []);

  const fetchThreads = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/threads");
      const data = await res.json();
      setThreads(data);
    } catch (error) {
      console.error("Failed to fetch threads:", error);
    }
  };

  const loadConversation = async (threadId: string) => {
    setCurrentThreadId(threadId);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
    try {
      const res = await fetch(`http://localhost:8000/api/threads/${threadId}`);
      const data = await res.json();
      setMessages(data);
    } catch (error) {
      console.error("Failed to load conversation:", error);
    }
  };

  const startNewChat = () => {
    setCurrentThreadId(crypto.randomUUID());
    setMessages([]);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMessage = input;
    setInput("");
    
    // FIX: Inject the user message AND a blank assistant message immediately
    setMessages((prev) => [
      ...prev, 
      { role: "user", content: userMessage },
      { role: "assistant", content: "" }
    ]);
    setIsStreaming(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, thread_id: currentThreadId }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunkValue = decoder.decode(value, { stream: true });
          const lines = chunkValue.split("\n\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const textChunk = line.replace("data: ", "").replace(/\\n/g, "\n");
              
              setMessages((prev) => {
                const updated = [...prev];
                const lastIndex = updated.length - 1;
                updated[lastIndex] = {
                  ...updated[lastIndex],
                  content: updated[lastIndex].content + textChunk
                };
                return updated;
              });
            }
          }
        }
      }
      fetchThreads(); 
    } catch (error) {
      console.error("Streaming error:", error);
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <>
      <Sidebar 
        threads={threads} 
        currentThreadId={currentThreadId} 
        loadConversation={loadConversation} 
        startNewChat={startNewChat}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />
      <ChatArea 
        messages={messages}
        input={input}
        setInput={setInput}
        sendMessage={sendMessage}
        isStreaming={isStreaming}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
      />
    </>
  );
}