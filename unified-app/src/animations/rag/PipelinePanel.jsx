import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { Database, FileText, Split, Hash, Search, Upload, Play, RotateCcw, CheckCircle2 } from 'lucide-react';

export default function PipelinePanel() {
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [documents, setDocuments] = useState([
        { id: 1, title: 'Q3 Report', content: 'Revenue increased by 15% to $4.2B. Operating margin improved to 23%.', color: 'bg-blue-100' },
        { id: 2, title: 'Product Manual', content: 'The XR-500 supports WiFi 6 and Bluetooth 5.2. Battery life is 12 hours.', color: 'bg-green-100' },
        { id: 3, title: 'HR Policy', content: 'Remote work is permitted 3 days per week. PTO accrues at 1.5 days/month.', color: 'bg-purple-100' },
    ]);
    const [chunks, setChunks] = useState([]);
    const [embeddings, setEmbeddings] = useState([]);
    const [indexed, setIndexed] = useState(false);
    const animRef = useRef(null);

    const steps = [
        {
            id: 0,
            title: '1. Document Collection',
            icon: Upload,
            description: 'Gather documents from various sources: PDFs, databases, APIs, web pages, etc.',
        },
        {
            id: 1,
            title: '2. Chunking',
            icon: Split,
            description: 'Split documents into smaller chunks (typically 256-512 tokens) with optional overlap.',
        },
        {
            id: 2,
            title: '3. Embedding',
            icon: Hash,
            description: 'Convert each chunk into a dense vector using an embedding model (e.g., text-embedding-3-small).',
        },
        {
            id: 3,
            title: '4. Indexing',
            icon: Database,
            description: 'Store embeddings in a vector database (Pinecone, Chroma, FAISS) for fast similarity search.',
        },
    ];

    const chunkDocument = () => {
        const newChunks = [];
        documents.forEach(doc => {
            const sentences = doc.content.split('. ');
            sentences.forEach((sent, i) => {
                newChunks.push({
                    id: `${doc.id}-${i}`,
                    docId: doc.id,
                    title: doc.title,
                    text: sent + (sent.endsWith('.') ? '' : '.'),
                    color: doc.color,
                });
            });
        });
        setChunks(newChunks);
    };

    const generateEmbeddings = () => {
        const newEmbeddings = chunks.map(chunk => ({
            ...chunk,
            vector: Array.from({ length: 5 }, () => (Math.random() * 2 - 1).toFixed(2)),
        }));
        setEmbeddings(newEmbeddings);
    };

    useEffect(() => {
        if (isPlaying && currentStep < 4) {
            const timer = setTimeout(() => {
                if (currentStep === 1) chunkDocument();
                if (currentStep === 2) generateEmbeddings();
                if (currentStep === 3) setIndexed(true);
                
                if (currentStep < 3) {
                    setCurrentStep(s => s + 1);
                } else {
                    setIsPlaying(false);
                }
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [isPlaying, currentStep]);

    const reset = () => {
        setCurrentStep(0);
        setIsPlaying(false);
        setChunks([]);
        setEmbeddings([]);
        setIndexed(false);
    };

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold text-indigo-900 mb-2">RAG Indexing Pipeline</h2>
                    <p className="text-slate-800 dark:text-slate-600">Watch how documents are prepared for retrieval</p>
                </div>

                {/* Controls */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(true)}
                        disabled={isPlaying || currentStep === 4}
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50"
                    >
                        <Play size={18} />
                        {currentStep === 0 ? 'Start Pipeline' : 'Continue'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                </div>

                {/* Step Progress */}
                <div className="flex justify-between items-center mb-8 px-4">
                    {steps.map((step, i) => (
                        <React.Fragment key={step.id}>
                            <div className={`flex flex-col items-center ${i <= currentStep ? 'text-indigo-600' : 'text-slate-300'}`}>
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 transition-all ${
                                    i < currentStep ? 'bg-green-500 text-white' :
                                    i === currentStep ? 'bg-indigo-500 text-white animate-pulse' :
                                    'bg-slate-200'
                                }`}>
                                    {i < currentStep ? <CheckCircle2 size={20} /> : <step.icon size={20} />}
                                </div>
                                <span className="text-xs font-medium text-center max-w-20">{step.title}</span>
                            </div>
                            {i < steps.length - 1 && (
                                <div className={`flex-1 h-1 mx-2 rounded ${i < currentStep ? 'bg-green-500' : 'bg-slate-200'}`} />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                {/* Current Step Description */}
                <div className="bg-indigo-50 rounded-xl p-4 mb-6 text-center border border-indigo-200">
                    <p className="text-indigo-800">{steps[Math.min(currentStep, 3)].description}</p>
                </div>

                {/* Visualization Area */}
                <div className="grid grid-cols-3 gap-4 min-h-96">
                    {/* Documents Column */}
                    <div className="bg-slate-50 rounded-xl p-4 border">
                        <h3 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
                            <FileText size={18} className="text-blue-500" />
                            Documents
                        </h3>
                        <div className="space-y-2">
                            {documents.map(doc => (
                                <div
                                    key={doc.id}
                                    className={`${doc.color} p-3 rounded-lg border transition-all ${
                                        currentStep >= 0 ? 'opacity-100' : 'opacity-50'
                                    }`}
                                >
                                    <div className="font-bold text-sm text-slate-800">{doc.title}</div>
                                    <div className="text-xs text-slate-800 dark:text-slate-600 mt-1">{doc.content}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Chunks Column */}
                    <div className="bg-slate-50 rounded-xl p-4 border">
                        <h3 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
                            <Split size={18} className="text-green-500" />
                            Chunks
                        </h3>
                        <div className="space-y-2 max-h-80 overflow-y-auto">
                            {currentStep >= 1 && chunks.map((chunk, i) => (
                                <div
                                    key={chunk.id}
                                    className={`${chunk.color} p-2 rounded-lg border text-xs transition-all animate-fadeIn`}
                                    style={{ animationDelay: `${i * 100}ms` }}
                                >
                                    <div className="font-semibold text-slate-700">{chunk.title} - Chunk {i + 1}</div>
                                    <div className="text-slate-800 dark:text-slate-600 mt-1">{chunk.text}</div>
                                </div>
                            ))}
                            {currentStep < 1 && (
                                <div className="text-slate-800 dark:text-sm italic p-4 text-center">
                                    Documents will be split into chunks...
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Embeddings/Index Column */}
                    <div className="bg-slate-50 rounded-xl p-4 border">
                        <h3 className="font-bold text-slate-800 mb-3 flex items-center gap-2">
                            <Database size={18} className="text-purple-500" />
                            Vector Index
                        </h3>
                        <div className="space-y-2 max-h-80 overflow-y-auto">
                            {currentStep >= 2 && embeddings.map((emb, i) => (
                                <div
                                    key={emb.id}
                                    className={`bg-purple-50 p-2 rounded-lg border transition-all ${
                                        indexed ? 'border-purple-300' : 'border-slate-200'
                                    }`}
                                >
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-xs font-semibold text-purple-700">
                                            {emb.title} #{i + 1}
                                        </span>
                                        {indexed && (
                                            <CheckCircle2 size={14} className="text-green-500" />
                                        )}
                                    </div>
                                    <div className="font-mono text-xs text-purple-600 bg-purple-100 p-1 rounded">
                                        [{emb.vector.join(', ')}]
                                    </div>
                                </div>
                            ))}
                            {currentStep < 2 && (
                                <div className="text-slate-800 dark:text-sm italic p-4 text-center">
                                    Embeddings will appear here...
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Chunking Strategy Info */}
                <div className="mt-6 bg-amber-50 rounded-xl p-4 border border-amber-200">
                    <h4 className="font-bold text-amber-900 mb-2">💡 Chunking Strategies</h4>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                            <span className="font-medium text-amber-800">Fixed Size:</span>
                            <p className="text-xs">Split every N tokens (simple but may break sentences)</p>
                        </div>
                        <div>
                            <span className="font-medium text-amber-800">Sentence-Based:</span>
                            <p className="text-xs">Split at sentence boundaries (preserves meaning)</p>
                        </div>
                        <div>
                            <span className="font-medium text-amber-800">Semantic:</span>
                            <p className="text-xs">Use embeddings to find natural breakpoints</p>
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
                    animation: fadeIn 0.3s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
