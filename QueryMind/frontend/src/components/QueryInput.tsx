import React, { useState } from "react";
import { SendIcon } from "lucide-react";

interface QueryInputProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

export const QueryInput: React.FC<QueryInputProps> = ({ onSubmit, loading }) => {
  const [query, setQuery] = useState("");

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !loading) {
        onSubmit(query.trim());
      }
    }
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto mb-6">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about your data..."
        disabled={loading}
        className="w-full bg-gray-800 text-gray-100 placeholder-gray-400 border border-gray-700 rounded-xl px-4 pt-3 pb-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none shadow-lg transition-all"
        rows={3}
      />
      <button
        type="button"
        onClick={() => {
          if (query.trim() && !loading) onSubmit(query.trim());
        }}
        disabled={loading || !query.trim()}
        className="absolute bottom-3 right-3 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg text-white transition-colors"
      >
        <SendIcon size={20} />
      </button>
      <div className="absolute bottom-3 left-4 text-xs text-gray-500">
        Press <kbd className="bg-gray-700 px-1 rounded">Enter</kbd> to submit, <kbd className="bg-gray-700 px-1 rounded">Shift+Enter</kbd> for newline
      </div>
    </div>
  );
};
