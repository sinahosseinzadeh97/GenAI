import React from "react";
import { QueryResult } from "../types";

interface ResultTableProps {
  result: QueryResult;
}

export const ResultTable: React.FC<ResultTableProps> = ({ result }) => {
  if (!result || !result.columns || !result.rows) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mb-6">
      <div className="flex justify-between items-center mb-2 px-2">
        <span className="text-sm text-gray-400">
          Showing {result.row_count} row{result.row_count !== 1 ? 's' : ''}
        </span>
        {result.from_cache && (
          <span className="px-2 py-0.5 bg-green-900 text-green-300 text-xs font-semibold rounded-full">
            Served from Cache
          </span>
        )}
      </div>
      <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-x-auto shadow-sm">
        <table className="w-full text-left text-sm text-gray-300">
          <thead className="text-xs text-gray-400 uppercase bg-gray-900/50">
            <tr>
              {result.columns.map((col, idx) => (
                <th key={idx} className="px-6 py-3 font-semibold tracking-wider">
                  {col.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, idx) => (
              <tr key={idx} className="border-b border-gray-700/50 hover:bg-gray-700/50 transition-colors">
                {result.columns.map((col, cIdx) => (
                  <td key={cIdx} className="px-6 py-4 whitespace-nowrap">
                    {row[col.name] !== null ? String(row[col.name]) : <span className="text-gray-500 italic">NULL</span>}
                  </td>
                ))}
              </tr>
            ))}
            {result.rows.length === 0 && (
              <tr>
                <td colSpan={result.columns.length} className="px-6 py-8 text-center text-gray-500">
                  No results found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
