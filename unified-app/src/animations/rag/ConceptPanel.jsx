import React, { useState, useEffect } from 'react';
import { Database, Search, MessageSquare, ArrowRight, Lightbulb, Brain, FileText, AlertTriangle } from 'lucide-react';

export default function ConceptPanel() {
    const [showComparison, setShowComparison] = useState('without');
    const [animStep, setAnimStep] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            setAnimStep(s => (s + 1) % 3);
        }, 1500);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="p-8 h-full">
            <div className="max-w-4xl mx-auto">
                {/* Introduction */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-indigo-900 mb-4">What is RAG?</h2>
                    <p className="text-lg text-slate-700 leading-relaxed max-w-2xl mx-auto">
                        <strong>Retrieval-Augmented Generation</strong> combines the power of LLMs with 
                        external knowledge retrieval, allowing models to access up-to-date, accurate information 
                        beyond their training data.
                    </p>
                </div>

                {/* The Problem */}
                <div className="bg-red-50 rounded-xl p-6 mb-8 border border-red-200">
                    <div className="flex items-start gap-4">
                        <AlertTriangle className="text-red-500 flex-shrink-0 mt-1" size={28} />
                        <div>
                            <h3 className="text-xl font-bold text-red-900 mb-2">The Problem with Standard LLMs</h3>
                            <ul className="text-red-800 space-y-2">
                                <li>• <strong>Knowledge Cutoff:</strong> Training data has a date limit (e.g., GPT-4: April 2023)</li>
                                <li>• <strong>Hallucinations:</strong> Models may generate plausible but incorrect information</li>
                                <li>• <strong>No Source Attribution:</strong> Can't cite where information comes from</li>
                                <li>• <strong>Domain-Specific Knowledge:</strong> May lack expertise in specialized areas</li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* The Solution */}
                <div className="bg-green-50 rounded-xl p-6 mb-8 border border-green-200">
                    <div className="flex items-start gap-4">
                        <Database className="text-green-500 flex-shrink-0 mt-1" size={28} />
                        <div>
                            <h3 className="text-xl font-bold text-green-900 mb-2">RAG to the Rescue!</h3>
                            <p className="text-green-800 mb-4">
                                RAG augments the LLM with a retrieval system that fetches relevant documents 
                                from an external knowledge base before generating a response.
                            </p>
                            <div className="grid grid-cols-3 gap-3 text-center">
                                <div className={`bg-white p-3 rounded-lg border transition-all ${animStep === 0 ? 'ring-2 ring-green-500 scale-105' : ''}`}>
                                    <Search className="mx-auto text-green-600 mb-1" size={24} />
                                    <div className="text-sm font-medium text-green-800">1. Retrieve</div>
                                </div>
                                <div className={`bg-white p-3 rounded-lg border transition-all ${animStep === 1 ? 'ring-2 ring-green-500 scale-105' : ''}`}>
                                    <FileText className="mx-auto text-green-600 mb-1" size={24} />
                                    <div className="text-sm font-medium text-green-800">2. Augment</div>
                                </div>
                                <div className={`bg-white p-3 rounded-lg border transition-all ${animStep === 2 ? 'ring-2 ring-green-500 scale-105' : ''}`}>
                                    <MessageSquare className="mx-auto text-green-600 mb-1" size={24} />
                                    <div className="text-sm font-medium text-green-800">3. Generate</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Interactive Comparison */}
                <div className="bg-slate-50 rounded-xl p-6 mb-8">
                    <h3 className="text-xl font-bold text-slate-800 mb-4 text-center">See the Difference</h3>
                    
                    <div className="flex justify-center gap-4 mb-6">
                        <button
                            onClick={() => setShowComparison('without')}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                showComparison === 'without'
                                    ? 'bg-red-500 text-white'
                                    : 'bg-white border hover:bg-slate-100'
                            }`}
                        >
                            ❌ Without RAG
                        </button>
                        <button
                            onClick={() => setShowComparison('with')}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                                showComparison === 'with'
                                    ? 'bg-green-500 text-white'
                                    : 'bg-white border hover:bg-slate-100'
                            }`}
                        >
                            ✅ With RAG
                        </button>
                    </div>

                    <div className="bg-white rounded-lg p-4 border">
                        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                            <span className="text-sm text-blue-600 font-medium">User Question:</span>
                            <p className="text-blue-800 font-medium">"What were the Q3 2024 revenue figures for Acme Corp?"</p>
                        </div>

                        {showComparison === 'without' ? (
                            <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                                <div className="flex items-start gap-3">
                                    <Brain className="text-red-500 flex-shrink-0 mt-1" size={24} />
                                    <div>
                                        <div className="text-sm text-red-600 mb-1">Standard LLM Response:</div>
                                        <p className="text-red-800">
                                            "I don't have specific information about Acme Corp's Q3 2024 revenue 
                                            as my training data only goes up to April 2023. However, based on 
                                            general market trends, similar companies typically see..."
                                        </p>
                                        <div className="mt-2 text-xs text-red-600 italic">
                                            ⚠️ May hallucinate or provide outdated information
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                                    <div className="text-sm text-green-600 mb-1">📚 Retrieved Context:</div>
                                    <p className="text-green-800 text-sm">
                                        "Acme Corp Q3 2024 Earnings Report: Revenue of $4.2B, up 15% YoY. 
                                        Operating margin improved to 23%..." [Source: earnings_q3_2024.pdf]
                                    </p>
                                </div>
                                <div className="p-3 bg-green-100 rounded-lg border border-green-300">
                                    <div className="flex items-start gap-3">
                                        <Brain className="text-green-500 flex-shrink-0 mt-1" size={24} />
                                        <div>
                                            <div className="text-sm text-green-600 mb-1">RAG-Enhanced Response:</div>
                                            <p className="text-green-800">
                                                "According to the Q3 2024 earnings report, Acme Corp reported 
                                                revenue of $4.2 billion, representing a 15% increase year-over-year. 
                                                The operating margin also improved to 23%."
                                            </p>
                                            <div className="mt-2 text-xs text-green-600">
                                                ✅ Accurate, up-to-date, and citable
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Use Cases */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                    {[
                        { icon: '📚', title: 'Documentation', desc: 'Query technical docs' },
                        { icon: '💼', title: 'Enterprise', desc: 'Internal knowledge bases' },
                        { icon: '📰', title: 'News', desc: 'Real-time information' },
                        { icon: '🔬', title: 'Research', desc: 'Scientific papers' },
                    ].map((use, i) => (
                        <div key={i} className="bg-white rounded-lg p-4 border text-center hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">{use.icon}</div>
                            <div className="font-bold text-slate-800">{use.title}</div>
                            <div className="text-xs text-slate-600">{use.desc}</div>
                        </div>
                    ))}
                </div>

                {/* Key Insight */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 flex items-start gap-3">
                    <Lightbulb className="text-amber-600 flex-shrink-0 mt-1" size={24} />
                    <div>
                        <h4 className="font-bold text-amber-900 mb-1">Key Insight</h4>
                        <p className="text-amber-800 text-sm">
                            RAG doesn't require retraining the LLM! It works by providing relevant context 
                            at inference time. This makes it much more cost-effective than fine-tuning and 
                            allows easy updates to the knowledge base.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
