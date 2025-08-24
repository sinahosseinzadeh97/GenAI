export type UploadResponse = { document_id: number; workflow_id: number };
export type Source = { document_id: number; title?: string | null; chunk: string };
export type QueryResponse = { answer: string; sources: Source[]; laws: Source[]; workflow_id?: number };
export type Workflow = { id: number; type: string; status: string; payload?: string; result?: string };