"use client";

import { useState, useEffect } from "react";
import Sidebar from "./Sidebar";
import ChatArea from "./ChatArea";
import ApprovalModal from "./ApprovalModal";
import DocumentManager from "./DocumentManager";

export type Message = { 
  role: "user" | "assistant"; 
  content: string;
};

export type Thread = { 
  thread_id: string; 
  title: string; 
};

export type PendingApproval = {
  type: string;
  symbol: string;
  quantity: number;
  message: string;
};

const API_BASE = "http://localhost:8000";

export default function ChatClient() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [isDocManagerOpen, setIsDocManagerOpen] = useState(false);

  useEffect(() => {
    fetchThreads();
    startNewChat();
  }, []);

  const fetchThreads = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/threads`);
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
      const res = await fetch(`${API_BASE}/api/threads/${threadId}`);
      const data = await res.json();
      setMessages(data);
      checkForApprovals(threadId);
    } catch (error) {
      console.error("Failed to load conversation:", error);
    }
  };

  const startNewChat = () => {
    setCurrentThreadId(crypto.randomUUID());
    setMessages([]);
    setPendingApproval(null);
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  const deleteChat = async (threadId: string) => {
    try {
      await fetch(`${API_BASE}/api/threads/${threadId}`, {
        method: 'DELETE'
      });
      fetchThreads();
      if (currentThreadId === threadId) {
        startNewChat();
      }
    } catch (error) {
      console.error("Failed to delete thread:", error);
    }
  };

  const checkForApprovals = async (threadId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/interrupts/${threadId}`);
      const data = await res.json();
      
      if (data.has_interrupt && data.interrupts && data.interrupts.length > 0) {
        const approval = data.interrupts[0].value;
        setPendingApproval(approval);
      }
    } catch (error) {
      console.error("Failed to check approvals:", error);
    }
  };

  const handleApproval = async (approved: boolean) => {
    if (!pendingApproval) return;
    
    try {
      await fetch(`${API_BASE}/api/chat/approval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: currentThreadId,
          approved
        })
      });
      
      setPendingApproval(null);
      setTimeout(() => loadConversation(currentThreadId), 500);
    } catch (error) {
      console.error("Approval failed:", error);
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMessage = input;
    setInput("");
    
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage },
      { role: "assistant", content: "" }
    ]);
    setIsStreaming(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: userMessage, 
          thread_id: currentThreadId 
        }),
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
              const textChunk = line.replace("data: ", "");
              
              if (textChunk.startsWith("[TOOL:")) {
                const toolName = textChunk.match(/\[TOOL: (.*)\]/)?.[1];
                if (toolName) {
                  setMessages((prev) => {
                    const updated = [...prev];
                    const lastIndex = updated.length - 1;
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      content: updated[lastIndex].content + `\n\n🔧 Using tool: ${toolName}\n\n`
                    };
                    return updated;
                  });
                }
              } else {
                const cleanChunk = textChunk.replace(/\\n/g, "\n");
                
                setMessages((prev) => {
                  const updated = [...prev];
                  const lastIndex = updated.length - 1;
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: updated[lastIndex].content + cleanChunk
                  };
                  return updated;
                });
              }
            }
          }
        }
      }
      
      fetchThreads();
      setTimeout(() => checkForApprovals(currentThreadId), 500);
      
    } catch (error) {
      console.error("Streaming error:", error);
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        updated[lastIndex] = {
          ...updated[lastIndex],
          content: "Error: Failed to get response from server."
        };
        return updated;
      });
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
        deleteChat={deleteChat}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
        onDocManagerClick={() => setIsDocManagerOpen(true)}
      />
      
      <ChatArea
        messages={messages}
        input={input}
        setInput={setInput}
        sendMessage={sendMessage}
        isStreaming={isStreaming}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
      />
      
      {pendingApproval && (
        <ApprovalModal
          approval={pendingApproval}
          onApprove={() => handleApproval(true)}
          onDecline={() => handleApproval(false)}
        />
      )}
      
      {isDocManagerOpen && (
        <DocumentManager
          onClose={() => setIsDocManagerOpen(false)}
        />
      )}
    </>
  );
}
