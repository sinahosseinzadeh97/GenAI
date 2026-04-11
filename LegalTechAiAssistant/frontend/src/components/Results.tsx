import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  DocumentTextIcon,
  ScaleIcon,
  ClipboardDocumentIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  StarIcon
} from '@heroicons/react/24/outline';
import { Source, Law } from '../types';
import clsx from 'clsx';

interface Props {
  answer?: string;
  sources?: Source[];
  laws?: Law[];
  draft?: string;
}

export default function Results({ answer, sources, laws, draft }: Props) {
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const [expandedLaws, setExpandedLaws] = useState<Set<number>>(new Set());

  const toggleSourceExpansion = (index: number) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSources(newExpanded);
  };

  const toggleLawExpansion = (index: number) => {
    const newExpanded = new Set(expandedLaws);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedLaws(newExpanded);
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  if (!answer && !sources?.length && !laws?.length && !draft) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="space-y-6"
    >
      {/* Answer Section */}
      {answer && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
              <DocumentTextIcon className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-800">AI Response</h3>
          </div>
          <div className="prose prose-slate max-w-none">
            <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{answer}</p>
          </div>
        </div>
      )}

      {/* Draft Section */}
      {draft && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-100 rounded-lg">
              <ClipboardDocumentIcon className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-800">Generated Draft</h3>
          </div>
          <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
            <pre className="text-slate-700 leading-relaxed whitespace-pre-wrap font-sans text-sm">
              {draft}
            </pre>
          </div>
        </div>
      )}

      {/* Sources Section */}
      {sources && sources.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-100 rounded-lg">
              <DocumentTextIcon className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-800">Document Sources</h3>
            <span className="px-2 py-1 bg-slate-100 text-slate-600 text-sm rounded-full">
              {sources.length} {sources.length === 1 ? 'source' : 'sources'}
            </span>
          </div>
          <div className="space-y-3">
            {sources.map((source, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border border-slate-200 rounded-lg p-4 hover:shadow-sm transition-shadow duration-200"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-medium text-slate-800">
                        {source.title || `Document ${source.document_id}`}
                      </h4>
                    </div>
                    <p className="text-slate-600 text-sm">
                      {expandedSources.has(index)
                        ? source.chunk
                        : `${source.chunk.slice(0, 200)}${source.chunk.length > 200 ? '...' : ''}`
                      }
                    </p>
                  </div>
                  {source.chunk.length > 200 && (
                    <button
                      onClick={() => toggleSourceExpansion(index)}
                      className="ml-2 p-1 text-slate-400 hover:text-slate-600 transition-colors"
                    >
                      {expandedSources.has(index) ? (
                        <ChevronUpIcon className="w-4 h-4" />
                      ) : (
                        <ChevronDownIcon className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Laws Section */}
      {laws && laws.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-amber-100 rounded-lg">
              <ScaleIcon className="w-6 h-6 text-amber-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-800">Legal References</h3>
            <span className="px-2 py-1 bg-slate-100 text-slate-600 text-sm rounded-full">
              {laws.length} {laws.length === 1 ? 'reference' : 'references'}
            </span>
          </div>
          <div className="space-y-3">
            {laws.map((law, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border border-slate-200 rounded-lg p-4 hover:shadow-sm transition-shadow duration-200"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-medium text-slate-800">{law.title || 'Legal Reference'}</h4>
                    </div>
                    <p className="text-slate-600 text-sm">
                      {expandedLaws.has(index)
                        ? law.chunk
                        : `${law.chunk.slice(0, 200)}${law.chunk.length > 200 ? '...' : ''}`
                      }
                    </p>
                  </div>
                  {law.chunk.length > 200 && (
                    <button
                      onClick={() => toggleLawExpansion(index)}
                      className="ml-2 p-1 text-slate-400 hover:text-slate-600 transition-colors"
                    >
                      {expandedLaws.has(index) ? (
                        <ChevronUpIcon className="w-4 h-4" />
                      ) : (
                        <ChevronDownIcon className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}