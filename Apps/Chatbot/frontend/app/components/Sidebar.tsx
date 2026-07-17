// components/Sidebar.tsx
import { Thread } from "./ChatClient";
import { MessageSquare, Plus, X, Code2 } from "lucide-react";

interface SidebarProps {
  threads: Thread[];
  currentThreadId: string;
  loadConversation: (id: string) => void;
  startNewChat: () => void;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

export default function Sidebar({ threads, currentThreadId, loadConversation, startNewChat, isOpen, setIsOpen }: SidebarProps) {
  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar Content */}
      <div className={`fixed inset-y-0 left-0 z-50 w-72 bg-gray-900 text-white flex flex-col transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${isOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Code2 size={20} className="text-white" />
            </div>
            <h1 className="text-lg font-bold tracking-wide">DevCraftYaseen</h1>
          </div>
          <button onClick={() => setIsOpen(false)} className="md:hidden text-gray-400 hover:text-white">
            <X size={24} />
          </button>
        </div>

        <div className="p-4">
          <button 
            onClick={startNewChat}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-xl transition duration-200"
          >
            <Plus size={20} />
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1">
          <h2 className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-3 px-2 mt-2">History</h2>
          {threads.map((thread) => (
            <button
              key={thread.thread_id}
              onClick={() => loadConversation(thread.thread_id)}
              className={`w-full flex items-center gap-3 text-left p-3 rounded-lg truncate transition duration-200 ${
                currentThreadId === thread.thread_id 
                  ? "bg-gray-800 text-blue-400 font-medium" 
                  : "hover:bg-gray-800/50 text-gray-300"
              }`}
            >
              <MessageSquare size={16} className="shrink-0" />
              <span className="truncate">{thread.title}</span>
            </button>
          ))}
        </div>
      </div>
    </>
  );
}