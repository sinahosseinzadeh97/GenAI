import { useState } from "react";
import { QueryResult, SSEEvent } from "../types";

export function useQueryStream() {
  const [status, setStatus] = useState<string>("");
  const [sql, setSql] = useState<string>("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const runQuery = async (nl_query: string, max_rows = 50) => {
    setLoading(true);
    setStatus("");
    setSql("");
    setResult(null);
    setError("");

    const response = await fetch("/query", {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "X-API-Key": import.meta.env.VITE_API_KEY || ""
      },
      body: JSON.stringify({ nl_query, max_rows })
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split("\n").filter(l => l.startsWith("data:"));

      for (const line of lines) {
        const json = line.replace("data:", "").trim();
        if (!json) continue;
        const event: SSEEvent = JSON.parse(json);

        if (event.type === "status") setStatus(event.content as string);
        if (event.type === "sql") setSql(event.content as string);
        if (event.type === "rows") setResult(event.content as QueryResult);
        if (event.type === "error") setError(event.content as string);
        if (event.type === "done") setLoading(false);
      }
    }
    setLoading(false);
  };

  return { runQuery, status, sql, result, error, loading };
}
