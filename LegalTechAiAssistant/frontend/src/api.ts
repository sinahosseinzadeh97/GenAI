const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function uploadDocument(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/documents`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as { document_id: number; workflow_id: number };
}

export async function askQuestion(question: string, action?: string) {
  const r = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, action })
  });
  if (!r.ok) throw new Error(await r.text());
  return (await r.json());
}

export async function getWorkflow(id: number) {
  const r = await fetch(`${API}/workflows/${id}`);
  if (!r.ok) throw new Error(await r.text());
  return (await r.json());
}