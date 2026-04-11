import React, { useEffect, useState } from "react";
import { DatabaseIcon } from "lucide-react";

export const SchemaPanel: React.FC = () => {
  const [schema, setSchema] = useState<string>("");

  useEffect(() => {
    const fetchSchema = async () => {
      try {
        const res = await fetch("/schema", {
          headers: {
            "X-API-Key": import.meta.env.VITE_API_KEY || ""
          }
        });
        const data = await res.json();
        setSchema(data.schema);
      } catch (e) {
        console.error("Failed to fetch schema", e);
      }
    };
    fetchSchema();
  }, []);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden shadow-sm flex flex-col h-[400px]">
      <div className="flex items-center px-4 py-3 bg-gray-900/50 border-b border-gray-700 gap-2">
        <DatabaseIcon size={16} className="text-gray-400" />
        <h3 className="font-semibold text-gray-200 text-sm tracking-wide uppercase">Database Schema</h3>
      </div>
      <div className="p-4 overflow-y-auto flex-1 bg-gray-900/20">
        {schema ? (
          <pre className="text-xs text-blue-200 font-mono whitespace-pre-wrap leading-relaxed">
            {schema}
          </pre>
        ) : (
          <div className="text-center text-gray-500 text-sm mt-8">
            Loading schema...
          </div>
        )}
      </div>
    </div>
  );
};
