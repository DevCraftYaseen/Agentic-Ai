// app/page.tsx
import ChatClient from "./components/ChatClient";

export default function Home() {
  return (
    <main className="flex h-screen bg-gray-50 overflow-hidden">
      <ChatClient />
    </main>
  );
}