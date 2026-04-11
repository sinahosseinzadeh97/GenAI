import React from "react";
import { QueryInsight } from "../types";
import { InfoIcon, LightbulbIcon, HelpCircleIcon } from "lucide-react";

interface InsightCardProps {
  insight: QueryInsight;
}

export const InsightCard: React.FC<InsightCardProps> = ({ insight }) => {
  if (!insight) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mb-8 bg-gray-800/80 border border-gray-700 rounded-xl p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-white mb-4">Query Insights</h3>
      
      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-blue-900/30 rounded-lg text-blue-400 mt-0.5">
            <InfoIcon size={18} />
          </div>
          <div>
            <h4 className="text-sm font-medium text-blue-200">Explanation</h4>
            <p className="text-gray-300 mt-1 text-sm">{insight.explanation}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <div className="p-2 bg-amber-900/30 rounded-lg text-amber-400 mt-0.5">
            <LightbulbIcon size={18} />
          </div>
          <div>
            <h4 className="text-sm font-medium text-amber-200">Business Insight</h4>
            <p className="text-gray-300 mt-1 text-sm">{insight.insight}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <div className="p-2 bg-purple-900/30 rounded-lg text-purple-400 mt-0.5">
            <HelpCircleIcon size={18} />
          </div>
          <div>
            <h4 className="text-sm font-medium text-purple-200">Suggested Follow-up</h4>
            <p className="text-gray-300 mt-1 text-sm">{insight.suggestion}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
