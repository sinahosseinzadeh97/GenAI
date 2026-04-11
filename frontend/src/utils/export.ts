const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";

export async function exportCSV(sessionId: string | null = null) {
  const params = sessionId ? `?session_id=${sessionId}` : "";
  const response = await fetch(`${API_BASE}/history/export/csv${params}`, {
    headers: { "X-API-Key": API_KEY }
  });
  
  if (!response.ok) throw new Error("Export failed");
  
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "query_history.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export async function exportJSON(sessionId: string | null = null) {
  const params = sessionId ? `?session_id=${sessionId}` : "";
  const response = await fetch(`${API_BASE}/history/export/json${params}`, {
    headers: { "X-API-Key": API_KEY }
  });
  
  if (!response.ok) throw new Error("Export failed");
  
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "query_history.json";
  a.click();
  URL.revokeObjectURL(url);
}
