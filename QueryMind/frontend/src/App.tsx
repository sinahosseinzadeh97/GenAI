import React, { useState } from "react";
import { useQueryStream } from "./hooks/useQueryStream";
import { QueryInput } from "./components/QueryInput";
import { StatusBar } from "./components/StatusBar";
import { SqlDisplay } from "./components/SqlDisplay";
import { ResultTable } from "./components/ResultTable";
import { InsightCard } from "./components/InsightCard";
import { HistoryPanel } from "./components/HistoryPanel";
import { SchemaPanel } from "./components/SchemaPanel";
import { RagPanel } from "./components/RagPanel";
import { AlertCircleIcon, BrainCircuitIcon, DatabaseIcon, FileSearchIcon } from "lucide-react";

export default function App() {
  const { runQuery, status, sql, result, error, loading } = useQueryStream();
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeMode, setActiveMode] = useState<"sql" | "rag">("sql");

  const handleSubmit = async (q: string) => {
    await runQuery(q);
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-[#0f1115] text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 bg-gray-900/50 sticky top-0 z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center gap-3">
          <div className="p-2 bg-blue-600/20 rounded-lg text-blue-400">
            <BrainCircuitIcon size={24} />
          </div>
          <div className="flex-1">
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
              QueryMind
            </h1>
            <p className="text-xs text-gray-400 font-medium tracking-wide">NATURAL LANGUAGE → SQL + RAG</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setActiveMode("sql")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeMode === "sql" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              <DatabaseIcon size={16} />
              SQL Mode
            </button>
            <button
              onClick={() => setActiveMode("rag")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeMode === "rag" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              <FileSearchIcon size={16} />
              RAG Mode
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex items-start gap-8">
        {activeMode === "sql" ? (
          <>
            <div className="w-1/4 flex-shrink-0 flex flex-col gap-6">
              <SchemaPanel />
              <HistoryPanel refreshTrigger={refreshTrigger} />
            </div>
            <div className="w-3/4 flex-grow flex flex-col pt-2">
              <QueryInput onSubmit={handleSubmit} loading={loading} />
              {error && (
                <div className="w-full max-w-4xl mx-auto mb-6 p-4 bg-red-900/30 border border-red-800/50 rounded-lg flex items-start gap-3">
                  <AlertCircleIcon className="text-red-400 mt-0.5 flex-shrink-0" size={20} />
                  <div className="text-red-200">{error}</div>
                </div>
              )}
              <StatusBar status={status} loading={loading} />
              <SqlDisplay sql={sql} />
              {result && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both">
                  <ResultTable result={result} />
                  {result.insight && <InsightCard insight={result.insight} />}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="w-full">
            <RagPanel />
          </div>
        )}
      </main>
    </div>
  );
}
