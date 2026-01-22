import React, { useState, useEffect } from 'react';
import { ArrowRight, X } from 'lucide-react';

export default function ChainPanel() {
    const [layers, setLayers] = useState(10);
    const [weight, setWeight] = useState(0.8);
    const [gradient, setGradient] = useState([]);

    useEffect(() => {
        // Simulate backpropagation
        // Initial gradient at output is 1.0
        const grads = [1.0];
        for (let i = 0; i < layers; i++) {
            grads.push(grads[grads.length - 1] * weight);
        }
        setGradient(grads);
    }, [layers, weight]);

    const finalGrad = gradient[gradient.length - 1];
    const isVanishing = Math.abs(finalGrad) < 0.001;
    const isExploding = Math.abs(finalGrad) > 1000;

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-red-400 mb-4">The Chain Rule</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    Gradients are calculated by <strong>multiplying</strong> many small numbers together.
                    <br />
                    If weights are even slightly off 1.0, the signal dies or explodes exponentially.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-6xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Network Parameters</h3>

                    <div className="space-y-6">
                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Network Depth (Layers)</label>
                                <span className="font-mono font-bold text-red-400">{layers}</span>
                            </div>
                            <input
                                type="range" min="2" max="50" step="1"
                                value={layers}
                                onChange={(e) => setLayers(parseInt(e.target.value))}
                                className="w-full accent-red-400"
                            />
                        </div>

                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Average Weight Magnitude</label>
                                <span className="font-mono font-bold text-red-400">{weight.toFixed(2)}</span>
                            </div>
                            <input
                                type="range" min="0.5" max="1.5" step="0.01"
                                value={weight}
                                onChange={(e) => setWeight(parseFloat(e.target.value))}
                                className="w-full accent-red-400"
                            />
                            <div className="flex justify-between text-xs text-slate-700 dark:text-slate-500 mt-1">
                                <span>Vanishing (&lt;1.0)</span>
                                <span>Stable (1.0)</span>
                                <span>Exploding (&gt;1.0)</span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 p-4 bg-slate-900 rounded-lg border border-slate-600">
                        <h4 className="text-sm font-bold text-slate-800 dark:text-slate-400 mb-2">Gradient at First Layer:</h4>
                        <div className={`text-3xl font-mono font-bold truncate ${isVanishing ? 'text-slate-600' : isExploding ? 'text-red-500' : 'text-green-400'
                            }`}>
                            {finalGrad.toExponential(4)}
                        </div>
                        <div className="mt-2 text-sm">
                            {isVanishing && <span className="text-slate-700 dark:text-slate-500">⚠️ Signal Vanished! The network stops learning.</span>}
                            {isExploding && <span className="text-red-400">⚠️ Signal Exploded! Weights become NaN.</span>}
                            {!isVanishing && !isExploding && <span className="text-green-500">✅ Signal is Healthy.</span>}
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 min-h-[400px] flex flex-col">
                    <h3 className="font-bold text-white mb-6">Backpropagation Flow</h3>

                    <div className="flex-1 flex flex-col-reverse gap-1 overflow-y-auto max-h-[500px] pr-2 custom-scrollbar">
                        {gradient.map((g, i) => {
                            // Visualizing magnitude
                            // Log scale for width
                            const width = Math.min(100, Math.max(1, Math.abs(g) * 100));
                            const color = Math.abs(g) > 10 ? 'bg-red-500' : Math.abs(g) < 0.1 ? 'bg-slate-700' : 'bg-green-500';

                            return (
                                <div key={i} className="flex items-center gap-4 group">
                                    <span className="text-xs font-mono text-slate-700 dark:text-slate-500 w-16 text-right">Layer {layers - i}</span>
                                    <div className="flex-1 h-8 bg-slate-900 rounded relative overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-300 ${color}`}
                                            style={{ width: `${Math.min(100, Math.abs(g) * 50)}%` }}
                                        />
                                        <span className="absolute inset-0 flex items-center pl-2 text-xs font-mono text-white mix-blend-difference">
                                            {g.toFixed(4)}
                                        </span>
                                    </div>
                                    <div className="text-xs text-slate-800 dark:text-slate-600 w-8">
                                        {i > 0 && <span className="flex items-center">× {weight}</span>}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    <div className="mt-4 text-center text-slate-800 dark:text-sm">
                        Gradient flows from Output (Bottom) to Input (Top)
                    </div>
                </div>
            </div>
        </div>
    );
}
