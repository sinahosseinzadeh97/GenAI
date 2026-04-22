// ── RAGForge Italia — Real API Client ────────────────────────────────────────
const HEADERS = { 'Content-Type': 'application/json', 'X-API-Key': '' };

async function req(path, opts = {}) {
  const res = await fetch(`/api${path}`, { ...opts, headers: { ...HEADERS, ...opts.headers } });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const checkHealth  = ()           => req('/health');
export const queryLex     = (query, col) => req('/lexreview/query',  { method:'POST', body: JSON.stringify({ query, collection_name: col }) });
export const checkVigenza = (norma, dt)  => req('/lexreview/vigenza',{ method:'POST', body: JSON.stringify({ norma, data_riferimento: dt }) });
export const risk231      = (settore, d) => req('/lexreview/231/risk-assessment', { method:'POST', body: JSON.stringify({ settore, descrizione_attivita: d }) });

export async function analyzeContratto(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/lexreview/contratto/analisi', { method: 'POST', headers: { 'X-API-Key': '' }, body: fd });
  if (!res.ok) { const b = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(b.detail || `HTTP ${res.status}`); }
  return res.json();
}
