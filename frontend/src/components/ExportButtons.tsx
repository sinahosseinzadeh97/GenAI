import React, { useState } from "react";
import { exportCSV, exportJSON } from "../utils/export";

interface ExportButtonsProps {
  sessionId?: string | null;
}

export const ExportButtons: React.FC<ExportButtonsProps> = ({ sessionId = null }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async (type: "csv" | "json") => {
    setLoading(true);
    setError(null);
    try {
      if (type === "csv") await exportCSV(sessionId);
      if (type === "json") await exportJSON(sessionId);
    } catch (e) {
      setError("Export failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="export-buttons">
      <button
        onClick={() => handleExport("csv")}
        disabled={loading}
        className="export-btn export-btn--csv"
      >
        {loading ? "Exporting..." : "⬇ Export CSV"}
      </button>
      <button
        onClick={() => handleExport("json")}
        disabled={loading}
        className="export-btn export-btn--json"
      >
        {loading ? "Exporting..." : "⬇ Export JSON"}
      </button>
      {error && <span className="export-error">{error}</span>}
    </div>
  );
};
