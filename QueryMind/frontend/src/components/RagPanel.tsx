import React, { useState } from "react";
import { UploadCloudIcon, SearchIcon, Loader2Icon, FileTextIcon } from "lucide-react";

export function RagPanel() {
  const [tab, setTab] = useState<"upload" | "search">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<any[]>([]);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/rag/ingest", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setUploadResult(data.message || JSON.stringify(data));
    } catch (e) {
      setUploadResult("Error uploading file.");
    }
    setUploading(false);
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setResults([]);
    try {
      const res = await fetch("http://localhost:8000/rag/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (e) {
      setResults([]);
    }
    setSearching(false);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex gap-2 border-b border-gray-700 pb-2">
        <button
          onClick={() => setTab("upload")}
          className={`px-4 py-2 rounded-t text-sm font-medium transition-colors ${
            tab === "upload" ? "bg-blue-600/20 text-blue-400 border-b-2 border-blue-500" : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <UploadCloudIcon size={16} className="inline mr-2" />
          Upload Contract
        </button>
        <button
          onClick={() => setTab("search")}
          className={`px-4 py-2 rounded-t text-sm font-medium transition-colors ${
            tab === "search" ? "bg-blue-600/20 text-blue-400 border-b-2 border-blue-500" : "text-gray-400 hover:text-gray-200"
          }`}
        >
          <SearchIcon size={16} className="inline mr-2" />
          Search Contracts
        </button>
      </div>

      {tab === "upload" && (
        <div className="flex flex-col gap-4">
          <div
            className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center cursor-pointer hover:border-blue-500 transition-colors"
            onClick={() => document.getElementById("pdf-upload")?.click()}
          >
            <UploadCloudIcon size={40} className="mx-auto text-gray-500 mb-3" />
            <p className="text-gray-400 text-sm">{file ? file.name : "Click to select a PDF contract"}</p>
            <input id="pdf-upload" type="file" accept="application/pdf" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </div>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2"
          >
            {uploading ? <Loader2Icon size={18} className="animate-spin" /> : <UploadCloudIcon size={18} />}
            {uploading ? "Ingesting..." : "Ingest PDF"}
          </button>
          {uploadResult && (
            <div className="p-4 bg-green-900/30 border border-green-700/50 rounded-lg text-green-300 text-sm">
              ✅ {uploadResult}
            </div>
          )}
        </div>
      )}

      {tab === "search" && (
        <div className="flex flex-col gap-4">
          <div className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search contracts... e.g. expiry date"
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={handleSearch}
              disabled={searching}
              className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-5 py-3 rounded-lg font-medium"
            >
              {searching ? <Loader2Icon size={18} className="animate-spin" /> : <SearchIcon size={18} />}
            </button>
          </div>
          {results.map((r, i) => (
            <div key={i} className="bg-gray-800/60 border border-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <FileTextIcon size={14} className="text-blue-400" />
                <span className="text-blue-400 text-xs font-medium">{r.filename}</span>
                <span className="text-gray-500 text-xs">— Page {r.page_number}</span>
              </div>
              <p className="text-gray-300 text-sm leading-relaxed">{r.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
