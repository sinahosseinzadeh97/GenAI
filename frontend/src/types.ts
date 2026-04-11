export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
}

export interface QueryInsight {
  explanation: string;
  insight: string;
  suggestion: string;
}

export interface QueryResult {
  columns: ColumnInfo[];
  rows: Record<string, unknown>[];
  row_count: number;
  from_cache: boolean;
  insight: QueryInsight | null;
}

export interface SSEEvent {
  type: "status" | "sql" | "rows" | "insight" | "error" | "done";
  content: unknown;
}
