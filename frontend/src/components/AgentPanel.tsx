import React, { useState, useRef, useEffect } from "react";
import { SendIcon, Loader2Icon, BotIcon, UserIcon, WrenchIcon } from "lucide-react";

interface Source {
  filename: string;
  page_number: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  tools_used?: string[];
  sources?: Source[];
}

export function AgentPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const sessionId = useRef("session_" + Date.now());

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/agent/chat", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-API-Key": import.meta.env.VITE_API_KEY || ""
        },
        body: JSON.stringify({ message: userMsg.content, session_id: sessionId.current }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer,
        tools_used: data.tools_used,
        sources: data.sources,
      }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "Error: could not reach the agent." }]);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-[75vh]">
      <div className="flex-1 overflow-y-auto flex flex-col gap-4 pb-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <BotIcon size={40} className="mx-auto mb-3 text-gray-600" />
            <p className="text-sm">Ask anything about your contracts</p>
            <p className="text-xs mt-2 text-gray-600">e.g. "When does the contract expire?" · "Who signed the agreement?" · "What are the payment terms?"</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="p-2 bg-blue-600/20 rounded-full h-fit mt-1">
                <BotIcon size={16} className="text-blue-400" />
              </div>
            )}
            <div className="max-w-[75%] flex flex-col gap-2">
              <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-tr-sm"
                  : "bg-gray-800 text-gray-100 rounded-tl-sm"
              }`}>
                {msg.content}
              </div>
              {msg.tools_used && msg.tools_used.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {msg.tools_used.map((t, j) => (
                    <span key={j} className="flex items-center gap-1 text-xs bg-gray-900 text-yellow-400 border border-yellow-900/40 px-2 py-1 rounded-full">
                      <WrenchIcon size={10} /> {t}
                    </span>
                  ))}
                </div>
              )}
              {msg.sources && msg.sources.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="text-xs bg-gray-900 text-blue-400 border border-blue-900/40 px-2 py-1 rounded-full">
                      📄 {s.filename} p.{s.page_number}
                    </span>
                  ))}
                </div>
              )}
            </div>
            {msg.role === "user" && (
              <div className="p-2 bg-gray-700 rounded-full h-fit mt-1">
                <UserIcon size={16} className="text-gray-300" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="p-2 bg-blue-600/20 rounded-full h-fit">
              <BotIcon size={16} className="text-blue-400" />
            </div>
            <div className="bg-gray-800 px-4 py-3 rounded-2xl rounded-tl-sm">
              <Loader2Icon size={16} className="animate-spin text-gray-400" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="flex gap-3 pt-4 border-t border-gray-800">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Ask about your contracts..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-5 py-3 rounded-xl transition-colors"
        >
          <SendIcon size={18} />
        </button>
      </div>
    </div>
  );
}
