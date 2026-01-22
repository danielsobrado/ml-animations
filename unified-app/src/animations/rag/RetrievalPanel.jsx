import React, { useState, useEffect, useRef } from 'react';
import { Search, Database, ArrowRight, Zap, FileText, Brain, Target } from 'lucide-react';

export default function RetrievalPanel() {
    const [query, setQuery] = useState('What is the battery life of XR-500?');
    const [searchStep, setSearchStep] = useState(0);
    const [topK, setTopK] = useState(3);
    const [isSearching, setIsSearching] = useState(false);
    const [showResponse, setShowResponse] = useState(false);

    // Simulated document chunks with embeddings
    const documentChunks = [
        { id: 1, text: 'The XR-500 supports WiFi 6 and Bluetooth 5.2.', source: 'Product Manual', relevance: 0.65 },
        { id: 2, text: 'Battery life is 12 hours with normal usage.', source: 'Product Manual', relevance: 0.95 },
        { id: 3, text: 'Remote work is permitted 3 days per week.', source: 'HR Policy', relevance: 0.15 },
        { id: 4, text: 'Revenue increased by 15% to $4.2B.', source: 'Q3 Report', relevance: 0.10 },
        { id: 5, text: 'The XR-500 has a 5000mAh battery capacity.', source: 'Product Manual', relevance: 0.88 },
        { id: 6, text: 'Operating margin improved to 23%.', source: 'Q3 Report', relevance: 0.08 },
        { id: 7, text: 'Fast charging reaches 80% in 30 minutes.', source: 'Product Manual', relevance: 0.82 },
        { id: 8, text: 'PTO accrues at 1.5 days per month.', source: 'HR Policy', relevance: 0.05 },
    ];

    const sortedByRelevance = [...documentChunks].sort((a, b) => b.relevance - a.relevance);
    const topResults = sortedByRelevance.slice(0, topK);

    const handleSearch = () => {
        setIsSearching(true);
        setSearchStep(1);
        setShowResponse(false);

        // Animate through steps
        setTimeout(() => setSearchStep(2), 1000);
        setTimeout(() => setSearchStep(3), 2000);
        setTimeout(() => setSearchStep(4), 3000);
        setTimeout(() => {
            setShowResponse(true);
            setIsSearching(false);
        }, 4000);
    };

    const getRelevanceColor = (relevance) => {
        if (relevance >= 0.8) return 'bg-green-100 border-green-300 text-green-800';
        if (relevance >= 0.5) return 'bg-yellow-100 border-yellow-300 text-yellow-800';
        return 'bg-slate-100 border-slate-200 text-slate-800 dark:text-slate-600';
    };

    const getRelevanceBarColor = (relevance) => {
        if (relevance >= 0.8) return 'bg-green-500';
        if (relevance >= 0.5) return 'bg-yellow-500';
        return 'bg-slate-300';
    };

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold text-indigo-900 mb-2">Retrieval in Action</h2>
                    <p className="text-slate-800 dark:text-slate-600">See how RAG retrieves relevant context and generates responses</p>
                </div>

                {/* Query Input */}
                <div className="bg-white rounded-xl p-4 border mb-6 shadow-sm">
                    <label className="block text-sm font-medium text-slate-700 mb-2">User Query:</label>
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                            placeholder="Ask a question..."
                        />
                        <select
                            value={topK}
                            onChange={(e) => setTopK(Number(e.target.value))}
                            className="px-3 py-2 border rounded-lg bg-white"
                        >
                            <option value={1}>Top 1</option>
                            <option value={2}>Top 2</option>
                            <option value={3}>Top 3</option>
                            <option value={5}>Top 5</option>
                        </select>
                        <button
                            onClick={handleSearch}
                            disabled={isSearching}
                            className="flex items-center gap-2 px-6 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50"
                        >
                            <Search size={18} />
                            Search
                        </button>
                    </div>
                </div>

                {/* Step Progress */}
                <div className="flex justify-center items-center gap-4 mb-6">
                    {[
                        { step: 1, label: 'Embed Query', icon: '🔢' },
                        { step: 2, label: 'Vector Search', icon: '🔍' },
                        { step: 3, label: 'Retrieve Top-K', icon: '📄' },
                        { step: 4, label: 'Generate', icon: '🤖' },
                    ].map((s, i) => (
                        <React.Fragment key={s.step}>
                            <div className={`flex flex-col items-center transition-all ${
                                searchStep >= s.step ? 'opacity-100 scale-100' : 'opacity-40 scale-90'
                            }`}>
                                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl mb-1 ${
                                    searchStep >= s.step ? 'bg-indigo-500' : 'bg-slate-200'
                                }`}>
                                    {s.icon}
                                </div>
                                <span className="text-xs font-medium text-slate-800 dark:text-slate-600">{s.label}</span>
                            </div>
                            {i < 3 && (
                                <ArrowRight className={`${searchStep > s.step ? 'text-indigo-500' : 'text-slate-300'}`} size={20} />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                {/* Main Visualization */}
                <div className="grid grid-cols-2 gap-6">
                    {/* Vector Database */}
                    <div className="bg-slate-50 rounded-xl p-4 border">
                        <h3 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
                            <Database size={18} className="text-purple-500" />
                            Vector Database
                        </h3>
                        <div className="space-y-2 max-h-80 overflow-y-auto">
                            {documentChunks.map((chunk, i) => {
                                const isRelevant = searchStep >= 2 && topResults.some(r => r.id === chunk.id);
                                return (
                                    <div
                                        key={chunk.id}
                                        className={`p-2 rounded-lg border transition-all duration-300 ${
                                            isRelevant
                                                ? 'ring-2 ring-indigo-500 ' + getRelevanceColor(chunk.relevance)
                                                : 'bg-white border-slate-200'
                                        }`}
                                    >
                                        <div className="flex justify-between items-start mb-1">
                                            <span className="text-xs font-medium text-slate-700 dark:text-slate-500">{chunk.source}</span>
                                            {searchStep >= 2 && (
                                                <span className={`text-xs font-bold ${
                                                    chunk.relevance >= 0.8 ? 'text-green-600' :
                                                    chunk.relevance >= 0.5 ? 'text-yellow-600' : 'text-slate-400'
                                                }`}>
                                                    {(chunk.relevance * 100).toFixed(0)}%
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-slate-700">{chunk.text}</p>
                                        {searchStep >= 2 && (
                                            <div className="mt-1 h-1 bg-slate-200 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full transition-all duration-500 ${getRelevanceBarColor(chunk.relevance)}`}
                                                    style={{ width: `${chunk.relevance * 100}%` }}
                                                />
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* RAG Process */}
                    <div className="space-y-4">
                        {/* Query Embedding */}
                        {searchStep >= 1 && (
                            <div className="bg-blue-50 rounded-xl p-4 border border-blue-200 animate-fadeIn">
                                <h4 className="font-bold text-blue-800 mb-2 flex items-center gap-2">
                                    <Target size={16} />
                                    Query Embedding
                                </h4>
                                <div className="font-mono text-xs bg-blue-100 p-2 rounded">
                                    [0.23, -0.15, 0.87, 0.42, -0.31, ...]
                                </div>
                            </div>
                        )}

                        {/* Retrieved Context */}
                        {searchStep >= 3 && (
                            <div className="bg-green-50 rounded-xl p-4 border border-green-200 animate-fadeIn">
                                <h4 className="font-bold text-green-800 mb-2 flex items-center gap-2">
                                    <FileText size={16} />
                                    Retrieved Context (Top {topK})
                                </h4>
                                <div className="space-y-2">
                                    {topResults.map((result, i) => (
                                        <div key={result.id} className="bg-white p-2 rounded border border-green-200 text-sm">
                                            <div className="flex justify-between">
                                                <span className="font-medium text-green-700">#{i + 1} {result.source}</span>
                                                <span className="text-green-600">{(result.relevance * 100).toFixed(0)}%</span>
                                            </div>
                                            <p className="text-xs mt-1">{result.text}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Generated Response */}
                        {showResponse && (
                            <div className="bg-purple-50 rounded-xl p-4 border border-purple-200 animate-fadeIn">
                                <h4 className="font-bold text-purple-800 mb-2 flex items-center gap-2">
                                    <Brain size={16} />
                                    Generated Response
                                </h4>
                                <div className="bg-white p-3 rounded border border-purple-200">
                                    <p className="text-slate-800">
                                        The XR-500 has a battery life of <strong>12 hours</strong> with normal usage. 
                                        It features a 5000mAh battery capacity and supports fast charging, 
                                        reaching 80% charge in just 30 minutes.
                                    </p>
                                    <div className="mt-2 text-xs text-purple-600">
                                        📚 Sources: Product Manual (3 chunks)
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Explanation */}
                <div className="mt-6 bg-amber-50 rounded-xl p-4 border border-amber-200">
                    <h4 className="font-bold text-amber-900 mb-2">⚡ How Vector Search Works</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <p className="text-amber-800">
                                <strong>Cosine Similarity:</strong> Measures the angle between vectors. 
                                Closer vectors have higher similarity (closer to 1.0).
                            </p>
                        </div>
                        <div>
                            <p className="text-amber-800">
                                <strong>Top-K Retrieval:</strong> Returns the K most similar chunks. 
                                Higher K = more context but may include less relevant info.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.4s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
