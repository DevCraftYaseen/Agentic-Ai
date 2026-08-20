"use client";

import { Thread } from "./ChatClient";
import { MessageSquare, Plus, X, Sparkles, FileText, Trash2 } from "lucide-react";
import { useState } from "react";

interface SidebarProps {
  threads: Thread[];
  currentThreadId: string;
  loadConversation: (id: string) => void;
  startNewChat: () => void;
  deleteChat: (id: string) => void;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  onDocManagerClick: () => void;
}

export default function Sidebar({ 
  threads, 
  currentThreadId, 
  loadConversation, 
  startNewChat, 
  deleteChat,
  isOpen, 
  setIsOpen,
  onDocManagerClick
}: SidebarProps) {
  const [hoveredThread, setHoveredThread] = useState<string | null>(null);

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside 
        className={`
          fixed md:static inset-y-0 left-0 z-50 
          w-80 bg-card border-r border-border
          flex flex-col shadow-2xl md:shadow-none
          transition-transform duration-200 ease-out
          ${isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}
      >
        {/* Header */}
        <div className="flex-shrink-0 p-6 border-b border-border">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center shadow-sm">
                <Sparkles className="w-5 h-5 text-primary-foreground" />
              </div>
              <h1 className="text-xl font-semibold text-foreground">AI Assistant</h1>
            </div>
            <button 
              onClick={() => setIsOpen(false)} 
              className="md:hidden p-2 hover:bg-accent rounded-lg text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close sidebar"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* New Chat Button */}
          <button 
            onClick={startNewChat}
            className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 px-4 rounded-xl transition-all shadow-sm hover:shadow-md active:scale-[0.98]"
          >
            <Plus className="w-5 h-5" />
            New Chat
          </button>
        </div>

        {/* Threads List */}
        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="text-xs text-muted-foreground font-semibold uppercase tracking-wider mb-3 px-3">
            Recent Chats
          </h2>
          
          {threads.length === 0 ? (
            <div className="text-center py-16 px-4">
              <div className="w-16 h-16 bg-muted rounded-2xl flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="w-8 h-8 text-muted-foreground" />
              </div>
              <p className="text-sm font-medium text-muted-foreground">No conversations yet</p>
              <p className="text-xs text-muted-foreground mt-2">Start a new chat to begin</p>
            </div>
          ) : (
            <div className="space-y-2">
              {threads.map((thread) => (
                <div
                  key={thread.thread_id}
                  className="relative group"
                  onMouseEnter={() => setHoveredThread(thread.thread_id)}
                  onMouseLeave={() => setHoveredThread(null)}
                >
                  <button
                    onClick={() => loadConversation(thread.thread_id)}
                    className={`
                      w-full flex items-center gap-3 text-left py-3 px-4 rounded-xl
                      transition-all duration-150 font-medium
                      ${currentThreadId === thread.thread_id 
                        ? "bg-primary text-primary-foreground shadow-sm" 
                        : "text-foreground hover:bg-accent"
                      }
                    `}
                  >
                    <MessageSquare className="w-4 h-4 shrink-0" />
                    <span className="truncate flex-1 text-sm">{thread.title}</span>
                  </button>
                  
                  {hoveredThread === thread.thread_id && currentThreadId !== thread.thread_id && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChat(thread.thread_id);
                      }}
                      className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-destructive hover:bg-destructive/90 text-destructive-foreground rounded-lg transition-all opacity-0 group-hover:opacity-100 shadow-sm"
                      title="Delete chat"
                      aria-label="Delete chat"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Documents Button */}
        <div className="flex-shrink-0 p-6 border-t border-border bg-muted/30">
          <button 
            onClick={onDocManagerClick}
            className="w-full flex items-center justify-center gap-2 bg-card hover:bg-accent text-foreground font-medium py-3 px-4 rounded-xl transition-all border border-border shadow-sm hover:shadow active:scale-[0.98]"
          >
            <FileText className="w-5 h-5" />
            Manage Documents
          </button>
        </div>
      </aside>
    </>
  );
}
