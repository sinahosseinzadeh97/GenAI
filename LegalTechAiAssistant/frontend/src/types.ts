export interface UploadResponse {
  document_id: string;
  workflow_id: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  laws: Law[];
  draft?: string;
  workflow_id: string;
}

export interface Source {
  document_id: number;
  title?: string;
  chunk: string;
}

export interface Law {
  document_id: number;
  title?: string;
  chunk: string;
}

export interface WorkflowStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  created_at: string;
  updated_at: string;
}

export type ActionType = 'email' | 'summary' | 'contract' | undefined;