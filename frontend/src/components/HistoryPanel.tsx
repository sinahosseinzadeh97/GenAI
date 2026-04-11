import React, { useEffect, useState } from "react";
import { MessageSquareIcon, TrashIcon } from "lucide-react";

interface Turn {
  role: string;
  nl_query: string | null;
  sql: string | null;
  result_summary: string | null;
}

interface HistoryPanelProps {
  refreshTrigger: number;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ refreshTrigger }) => {
  const [turns, setTurns] = useState<Turn[]>([]);

  const fetchHistory = async () => {
    try {
      const res = await fetch("/history", {
        headers: {
          "X-API-Key": import.meta.env.VITE_API_KEY || ""
        }
      });
      const data = await res.json();
      setTurns(data.turns);
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [refreshTrigger]);

  const clearHistory = async () => {
    try {
      await fetch("/history", {
        method: "DELETE",
        headers: {
          "X-API-Key": import.meta.env.VITE_API_KEY || ""
        }
      });
      setTurns([]);
    } catch (e) {
      console.error("Failed to clear history", e);
    }
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-sm flex flex-col h-[400px]">
      <div className="flex justify-between items-center px-4 py-3 bg-gray-900/50 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <MessageSquareIcon size={16} className="text-gray-400" />
          <h3 className="font-semibold text-gray-200 text-sm tracking-wide uppercase">Conversation History</h3>
        </div>
        {(turns.length > 0) && (
          <button 
            onClick={clearHistory}
            className="text-gray-500 hover:text-red-400 p-1 transition-colors rounded"
            title="Clear History"
          >
            <TrashIcon size={16} />
          </button>
        )}
      </div>
      
      <div className="p-4 overflow-y-auto flex-1 space-y-4">
        {turns.length === 0 ? (
          <div className="text-center text-gray-500 text-sm mt-8">
            No history yet.
          </div>
        ) : (
          turns.slice(-10).map((turn, i) => (
            <div key={i} className="flex flex-col gap-1 text-sm bg-gray-900/40 p-3 rounded-lg border border-gray-700/50">
              <div className="flex items-center justify-between mb-1">
                <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                  turn.role === 'user' ? 'bg-blue-900/40 text-blue-300' : 'bg-green-900/40 text-green-300'
                }`}>
                  {turn.role.toUpperCase()}
                </span>
              </div>
              {turn.nl_query && <div className="text-gray-200">{turn.nl_query}</div>}
              {turn.sql && turn.sql !== '[cache hit]' && (
                <div className="font-mono text-xs text-gray-500 bg-gray-900 p-1.5 rounded truncate">
                  {turn.sql}
                </div>
              )}
              {turn.result_summary && <div className="text-gray-400 italic text-xs">{turn.result_summary}</div>}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
