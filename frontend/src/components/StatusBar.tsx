import React from "react";
import { Loader2Icon } from "lucide-react";

interface StatusBarProps {
  status: string;
  loading: boolean;
}

export const StatusBar: React.FC<StatusBarProps> = ({ status, loading }) => {
  if (!status && !loading) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mb-6 flex items-center justify-center p-4 bg-gray-800 border border-gray-700 rounded-lg shadow-sm">
      {loading && <Loader2Icon className="animate-spin text-blue-500 mr-3" size={24} />}
      <span className="text-gray-200 font-medium">{status || "Processing..."}</span>
    </div>
  );
};
