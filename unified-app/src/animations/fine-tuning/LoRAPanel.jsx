import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Layers, ArrowRight, Plus, Equal, Lightbulb } from 'lucide-react';
import { gsap } from 'gsap';

export default function LoRAPanel() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [step, setStep] = useState(0);
    const [rank, setRank] = useState(8);
    const matrixRef = useRef(null);

    // Example dimensions
    const d = 768;  // Input dimension
    const k = 768;  // Output dimension

    const steps = [
        {
            title: 'Original Weight Matrix W',
            desc: `The original frozen weight matrix has shape ${d}×${k} = ${(d*k).toLocaleString()} parameters`,
        },
        {
            title: 'Low-Rank Decomposition',
            desc: `Instead of updating W directly, we learn two small matrices A (${d}×${rank}) and B (${rank}×${k})`,
        },
        {
            title: 'Compute ΔW = BA',
            desc: `The product BA has the same shape as W, but only requires ${d*rank + rank*k} = ${(d*rank + rank*k).toLocaleString()} parameters!`,
        },
        {
            title: 'Final Output: W + ΔW',
            desc: `During inference, we add the learned adaptation to the frozen weights. Can be merged for zero latency!`,
        },
    ];

    useEffect(() => {
        if (isPlaying && step < steps.length - 1) {
            const timer = setTimeout(() => setStep(s => s + 1), 2500);
            return () => clearTimeout(timer);
        } else if (step >= steps.length - 1) {
            setIsPlaying(false);
        }
    }, [isPlaying, step]);

    const reset = () => {
        setStep(0);
        setIsPlaying(false);
    };

    const paramReduction = ((d*k - (d*rank + rank*k)) / (d*k) * 100).toFixed(1);

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold text-purple-900 mb-2">LoRA: Low-Rank Adaptation</h2>
                    <p className="text-slate-600">Freeze the base model, learn small rank-decomposed matrices</p>
                </div>

                {/* Controls */}
                <div className="flex justify-center items-center gap-4 mb-6">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                    <div className="flex items-center gap-2 ml-4">
                        <span className="text-sm text-slate-600">Rank (r):</span>
                        <select
                            value={rank}
                            onChange={(e) => setRank(Number(e.target.value))}
                            className="px-3 py-1 border rounded bg-white"
                        >
                            <option value={4}>4</option>
                            <option value={8}>8</option>
                            <option value={16}>16</option>
                            <option value={32}>32</option>
                            <option value={64}>64</option>
                        </select>
                    </div>
                </div>

                {/* Step Indicator */}
                <div className="flex justify-center gap-2 mb-6">
                    {steps.map((s, i) => (
                        <button
                            key={i}
                            onClick={() => setStep(i)}
                            className={`px-3 py-1 rounded-full text-sm transition-all ${
                                i === step
                                    ? 'bg-purple-500 text-white'
                                    : i < step
                                        ? 'bg-green-100 text-green-700'
                                        : 'bg-slate-200 text-slate-500'
                            }`}
                        >
                            {i + 1}
                        </button>
                    ))}
                </div>

                {/* Current Step Description */}
                <div className="bg-purple-50 rounded-xl p-4 mb-6 text-center border border-purple-200">
                    <h3 className="font-bold text-purple-800 mb-1">{steps[step].title}</h3>
                    <p className="text-purple-700 text-sm">{steps[step].desc}</p>
                </div>

                {/* Visualization */}
                <div className="bg-white rounded-xl p-6 border shadow-sm mb-6" ref={matrixRef}>
                    <div className="flex items-center justify-center gap-4 flex-wrap">
                        {/* W Matrix - Always shown */}
                        <div className={`transition-all ${step > 0 ? 'opacity-50' : ''}`}>
                            <div className="text-center mb-2 font-bold text-slate-700">W (frozen)</div>
                            <div 
                                className="bg-gradient-to-br from-blue-200 to-blue-400 rounded-lg flex items-center justify-center"
                                style={{ width: 120, height: 120 }}
                            >
                                <span className="text-blue-800 font-mono text-xs">{d}×{k}</span>
                            </div>
                            <div className="text-center mt-1 text-xs text-slate-500">
                                {(d*k).toLocaleString()} params
                            </div>
                        </div>

                        {step >= 1 && (
                            <>
                                <Plus className="text-slate-400" size={24} />
                                
                                {/* ΔW Decomposition */}
                                <div className="flex items-center gap-2">
                                    {/* A Matrix */}
                                    <div className="animate-fadeIn">
                                        <div className="text-center mb-2 font-bold text-green-700">B</div>
                                        <div 
                                            className="bg-gradient-to-br from-green-200 to-green-400 rounded-lg flex items-center justify-center"
                                            style={{ width: 120, height: 30 + rank * 2 }}
                                        >
                                            <span className="text-green-800 font-mono text-xs">{d}×{rank}</span>
                                        </div>
                                        <div className="text-center mt-1 text-xs text-green-600">
                                            {(d*rank).toLocaleString()} params
                                        </div>
                                    </div>

                                    <span className="text-slate-400 font-bold">×</span>

                                    {/* B Matrix */}
                                    <div className="animate-fadeIn" style={{ animationDelay: '200ms' }}>
                                        <div className="text-center mb-2 font-bold text-orange-700">A</div>
                                        <div 
                                            className="bg-gradient-to-br from-orange-200 to-orange-400 rounded-lg flex items-center justify-center"
                                            style={{ width: 30 + rank * 2, height: 120 }}
                                        >
                                            <span className="text-orange-800 font-mono text-xs">{rank}×{k}</span>
                                        </div>
                                        <div className="text-center mt-1 text-xs text-orange-600">
                                            {(rank*k).toLocaleString()} params
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}

                        {step >= 2 && (
                            <>
                                <Equal className="text-slate-400" size={24} />
                                
                                {/* ΔW Result */}
                                <div className="animate-fadeIn">
                                    <div className="text-center mb-2 font-bold text-purple-700">ΔW</div>
                                    <div 
                                        className="bg-gradient-to-br from-purple-200 to-purple-400 rounded-lg flex items-center justify-center border-2 border-purple-500 border-dashed"
                                        style={{ width: 120, height: 120 }}
                                    >
                                        <span className="text-purple-800 font-mono text-xs">{d}×{k}</span>
                                    </div>
                                    <div className="text-center mt-1 text-xs text-purple-600">
                                        Low-rank update
                                    </div>
                                </div>
                            </>
                        )}
                    </div>

                    {step >= 3 && (
                        <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200 animate-fadeIn">
                            <div className="text-center">
                                <div className="text-lg font-bold text-green-800 mb-2">
                                    h = W·x + ΔW·x = W·x + B·A·x
                                </div>
                                <p className="text-sm text-green-700">
                                    The output combines the frozen pretrained knowledge with task-specific adaptation!
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 text-center">
                        <div className="text-2xl font-bold text-blue-600">{(d*k).toLocaleString()}</div>
                        <div className="text-sm text-blue-700">Original Params</div>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg border border-green-200 text-center">
                        <div className="text-2xl font-bold text-green-600">{(d*rank + rank*k).toLocaleString()}</div>
                        <div className="text-sm text-green-700">LoRA Params (r={rank})</div>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg border border-purple-200 text-center">
                        <div className="text-2xl font-bold text-purple-600">{paramReduction}%</div>
                        <div className="text-sm text-purple-700">Reduction</div>
                    </div>
                </div>

                {/* Key Formula */}
                <div className="bg-slate-800 rounded-xl p-6 mb-6 text-center">
                    <div className="text-slate-400 text-sm mb-2">The LoRA Formula</div>
                    <div className="text-white font-mono text-xl">
                        W' = W + α · BA
                    </div>
                    <div className="text-slate-400 text-sm mt-2">
                        where α is a scaling factor (typically α/r)
                    </div>
                </div>

                {/* Tips */}
                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200">
                    <h4 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
                        <Lightbulb size={18} />
                        Practical Tips
                    </h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-amber-800">
                        <div>
                            <strong>Rank Selection:</strong> Start with r=8 or r=16. Higher rank = more capacity but more params.
                        </div>
                        <div>
                            <strong>Which Layers:</strong> Usually apply LoRA to attention weights (Q, K, V projections).
                        </div>
                        <div>
                            <strong>Initialization:</strong> A initialized with small random values, B initialized to zero.
                        </div>
                        <div>
                            <strong>Merging:</strong> After training, LoRA weights can be merged into base model for fast inference.
                        </div>
                    </div>
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: scale(0.9); }
                    to { opacity: 1; transform: scale(1); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.4s ease-out forwards;
                }
            `}</style>
        </div>
    );
}
