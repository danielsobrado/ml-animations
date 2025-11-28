import React, { useState } from 'react';

export default function SurprisePanel() {
    const [prob, setProb] = useState(0.5);

    // Information (Surprise) = -log2(p)
    const info = -Math.log2(prob);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-pink-400 mb-4">What is a "Bit"?</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Information is measured in <strong>Surprise</strong>.
                    <br />
                    Rare events give us more information than common ones.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-center">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Event Probability</h3>

                    <div className="mb-8">
                        <div className="flex justify-between items-end mb-2">
                            <label className="text-sm text-slate-400">Probability (p)</label>
                            <span className="text-3xl font-mono font-bold text-pink-400">{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range" min="0.01" max="1" step="0.01"
                            value={prob}
                            onChange={(e) => setProb(parseFloat(e.target.value))}
                            className="w-full accent-pink-400"
                        />
                        <div className="flex justify-between text-xs text-slate-500 mt-2">
                            <span>Rare (1%)</span>
                            <span>Certain (100%)</span>
                        </div>
                    </div>

                    <div className="bg-slate-900 p-4 rounded-lg text-sm text-slate-300">
                        <p className="mb-2"><strong>Examples:</strong></p>
                        <ul className="list-disc list-inside space-y-1">
                            <li>p = 0.5 (Coin Flip): <strong>1 bit</strong></li>
                            <li>p = 0.125 (3 Coin Flips): <strong>3 bits</strong></li>
                            <li>p = 1.0 (Sun rises): <strong>0 bits</strong> (No surprise!)</li>
                        </ul>
                    </div>
                </div>

                {/* Visualization */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col items-center">
                    <h3 className="font-bold text-white mb-4">Information Content</h3>

                    <div className="relative w-48 h-64 bg-slate-900 rounded-lg overflow-hidden border border-slate-600 flex flex-col justify-end">
                        <div
                            className="w-full bg-pink-500 transition-all duration-300 flex items-center justify-center text-white font-bold text-2xl"
                            style={{ height: `${Math.min(100, info * 10)}%` }}
                        >
                        </div>
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="text-center">
                                <div className="text-4xl font-bold text-white drop-shadow-md">{info.toFixed(2)}</div>
                                <div className="text-sm text-slate-400">Bits</div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 font-mono text-lg text-slate-300">
                        I(x) = -log<sub>2</sub>({prob.toFixed(2)})
                    </div>
                </div>
            </div>
        </div>
    );
}
