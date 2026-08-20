"use client";

import { useEffect, useRef } from "react";
import { Message } from "./ChatClient";
import { Send, Menu, Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface ChatAreaProps {
  messages: Message[];
  input: string;
  setInput: (val: string) => void;
  sendMessage: (e: React.FormEvent) => void;
  isStreaming: boolean;
  toggleSidebar: () => void;
}

export default function ChatArea({ 
  messages, 
  input, 
  setInput, 
  sendMessage, 
  isStreaming, 
  toggleSidebar 
}: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Mobile Header */}
      <div className="md:hidden flex items-center p-4 border-b border-border bg-card shadow-sm">
        <button 
          onClick={toggleSidebar} 
          className="p-2 -ml-2 text-muted-foreground hover:text-foreground rounded-xl transition-colors hover:bg-accent"
          aria-label="Open sidebar"
        >
          <Menu className="w-6 h-6" />
        </button>
        <span className="ml-3 font-semibold text-foreground">AI Assistant</span>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 md:p-8">
        {messages.length === 0 ? (
          <div className="flex flex-col h-full items-center justify-center px-4 max-w-4xl mx-auto">
            <div className="bg-primary p-6 rounded-3xl mb-6 shadow-lg">
              <Bot className="w-12 h-12 text-primary-foreground" />
            </div>
            <div className="text-center space-y-3 mb-12">
              <h1 className="text-4xl font-bold text-foreground">Welcome to AI Assistant</h1>
              <p className="text-muted-foreground max-w-xl text-lg leading-relaxed">
                I can help you with web searches, calculations, stock prices, document analysis, and more!
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
              <div className="border-2 border-border hover:border-primary rounded-2xl p-6 transition-all cursor-pointer group hover:shadow-md bg-card">
                <div className="text-3xl mb-3">🔍</div>
                <p className="font-semibold text-foreground mb-1">Web Search</p>
                <p className="text-sm text-muted-foreground">Search for current information</p>
              </div>
              <div className="border-2 border-border hover:border-primary rounded-2xl p-6 transition-all cursor-pointer group hover:shadow-md bg-card">
                <div className="text-3xl mb-3">🧮</div>
                <p className="font-semibold text-foreground mb-1">Calculator</p>
                <p className="text-sm text-muted-foreground">Solve mathematical problems</p>
              </div>
              <div className="border-2 border-border hover:border-primary rounded-2xl p-6 transition-all cursor-pointer group hover:shadow-md bg-card">
                <div className="text-3xl mb-3">📈</div>
                <p className="font-semibold text-foreground mb-1">Stock Prices</p>
                <p className="text-sm text-muted-foreground">Get real-time market data</p>
              </div>
              <div className="border-2 border-border hover:border-primary rounded-2xl p-6 transition-all cursor-pointer group hover:shadow-md bg-card">
                <div className="text-3xl mb-3">📄</div>
                <p className="font-semibold text-foreground mb-1">Document Q&A</p>
                <p className="text-sm text-muted-foreground">Ask questions about PDFs</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`flex gap-4 animate-fadeIn ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" && (
                  <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shrink-0 shadow-sm">
                    <Bot className="w-5 h-5 text-primary-foreground" />
                  </div>
                )}

                <div className={`max-w-[85%] md:max-w-3xl p-4 rounded-2xl ${
                  msg.role === "user" 
                    ? "bg-primary text-primary-foreground rounded-tr-md shadow-md" 
                    : "bg-card text-card-foreground border border-border rounded-tl-md shadow-sm"
                }`}>
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  ) : (
                    <div className="prose prose-sm md:prose-base max-w-none">
                      {msg.content === "" && isStreaming ? (
                        <div className="flex space-x-2 h-6 items-center">
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                        </div>
                      ) : (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      )}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center shrink-0 shadow-sm">
                    <User className="w-5 h-5 text-muted-foreground" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-6 bg-card border-t border-border">
        <form onSubmit={sendMessage} className="max-w-4xl mx-auto flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(e);
              }
            }}
            disabled={isStreaming}
            placeholder="Ask me anything... (Shift+Enter for new line)"
            className="flex-1 max-h-32 min-h-[56px] text-foreground bg-background border-2 border-input focus:border-ring placeholder:text-muted-foreground rounded-2xl px-5 py-4 focus:outline-none disabled:opacity-50 resize-none transition-all shadow-sm"
            rows={1}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="h-[56px] w-[56px] flex items-center justify-center bg-primary hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed text-primary-foreground rounded-2xl transition-all shadow-md hover:shadow-lg shrink-0 active:scale-[0.98]"
            aria-label="Send message"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
        <p className="text-xs text-muted-foreground text-center mt-4">
          AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}
