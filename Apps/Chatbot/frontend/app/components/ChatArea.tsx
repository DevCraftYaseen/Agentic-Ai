// components/ChatArea.tsx
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

export default function ChatArea({ messages, input, setInput, sendMessage, isStreaming, toggleSidebar }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col h-full bg-white relative">
      {/* Mobile Header */}
      <div className="md:hidden flex items-center p-4 border-b border-gray-200 bg-white shadow-sm z-10">
        <button onClick={toggleSidebar} className="p-2 -ml-2 text-gray-600 hover:text-gray-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
          <Menu size={24} />
        </button>
        <span className="ml-2 font-bold text-gray-800">DevCraftYaseen Chat</span>
      </div>

      {/* Messages Array */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col h-full items-center justify-center text-gray-400 space-y-4">
            <Bot size={48} className="text-gray-300" />
            <p className="text-lg font-medium">How can I help you build today?</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              
              {/* AI Avatar */}
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center shrink-0 mt-1">
                  <Bot size={18} className="text-blue-600" />
                </div>
              )}

              {/* Chat Bubble */}
              <div className={`max-w-[85%] md:max-w-3xl p-4 rounded-2xl shadow-sm ${
                msg.role === "user" 
                  ? "bg-blue-600 text-white rounded-tr-sm" 
                  : "bg-gray-50 text-gray-800 border border-gray-100 rounded-tl-sm"
              }`}>
                {msg.role === "user" ? (
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                ) : (
                  <div className="prose prose-sm md:prose-base prose-blue max-w-none">
                    {/* FIX: If the message is empty and we are streaming, show the typing dots */}
                    {msg.content === "" && isStreaming ? (
                      <div className="flex space-x-1.5 h-6 items-center px-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      </div>
                    ) : (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    )}
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center shrink-0 mt-1">
                  <User size={18} className="text-gray-600" />
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Field */}
      <div className="p-4 md:p-6 bg-white border-t border-gray-100">
        <form onSubmit={sendMessage} className="max-w-4xl mx-auto relative flex items-end gap-2">
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
            placeholder="Type your message... (Shift+Enter for new line)"
            // FIX: Added text-gray-900, bg-white, and font-medium to make the text clearly visible
            className="flex-1 max-h-32 min-h-[56px] text-gray-900 font-medium bg-white border border-gray-300 placeholder-gray-400 rounded-xl px-4 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 resize-none shadow-sm"
            rows={1}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="h-[56px] w-[56px] flex items-center justify-center bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl font-semibold transition duration-200 shadow-sm shrink-0"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
}