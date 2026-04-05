import React from "react";
import { CopyIcon } from "lucide-react";

interface SqlDisplayProps {
  sql: string;
}

export const SqlDisplay: React.FC<SqlDisplayProps> = ({ sql }) => {
  if (!sql) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(sql);
  };

  return (
    <div className="w-full max-w-4xl mx-auto mb-6 bg-[#1e1e1e] border border-gray-700 rounded-lg overflow-hidden shadow-sm">
      <div className="flex justify-between items-center px-4 py-2 bg-gray-800 border-b border-gray-700">
        <span className="text-sm font-semibold text-gray-400">Generated SQL</span>
        <button
          onClick={handleCopy}
          className="text-gray-400 hover:text-white transition-colors"
          title="Copy SQL"
        >
          <CopyIcon size={16} />
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm text-blue-300 font-mono">
          <code>{sql}</code>
        </pre>
      </div>
    </div>
  );
};
