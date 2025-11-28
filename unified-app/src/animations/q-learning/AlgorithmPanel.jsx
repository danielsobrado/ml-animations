import React, { useState } from 'react';

export default function AlgorithmPanel() {
    // Simple scenario: State A -> State B
    const [qOld, setQOld] = useState(2.0);
    const [reward, setReward] = useState(1.0);
    const [qNextMax, setQNextMax] = useState(5.0);
    const [alpha, setAlpha] = useState(0.5); // Learning Rate
    const [gamma, setGamma] = useState(0.9); // Discount Factor

    // Bellman Calculation
    const target = reward + gamma * qNextMax;
    const tdError = target - qOld;
    const qNew = qOld + alpha * tdError;

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-sky-400 mb-4">The Bellman Update</h2>
                <div className="bg-slate-800 p-4 rounded-lg font-mono text-sm inline-block shadow-lg border border-slate-700">
                    Q(s,a) ← Q(s,a) + <span className="text-purple-400">α</span> [
                    <span className="text-green-400">R</span> +
                    <span className="text-orange-400">γ</span> max Q(s',a') - Q(s,a) ]
                </div>
            </div>

            <div className="grid lg:grid-cols-2 gap-12 w-full max-w-6xl">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 space-y-6">
                    <h3 className="font-bold text-white border-b border-slate-700 pb-2">Parameters</h3>

                    <div>
                        <label className="text-sm text-slate-400 block mb-1">Old Q-Value (Q(s,a))</label>
                        <input type="number" value={qOld} onChange={e => setQOld(parseFloat(e.target.value))} className="w-full bg-slate-900 border border-slate-600 rounded p-2 text-white" />
                    </div>

                    <div>
                        <label className="text-sm text-green-400 block mb-1">Reward (R)</label>
                        <input type="number" value={reward} onChange={e => setReward(parseFloat(e.target.value))} className="w-full bg-slate-900 border border-green-900/50 rounded p-2 text-white" />
                    </div>

                    <div>
                        <label className="text-sm text-blue-400 block mb-1">Max Future Q (max Q(s',a'))</label>
                        <input type="number" value={qNextMax} onChange={e => setQNextMax(parseFloat(e.target.value))} className="w-full bg-slate-900 border border-blue-900/50 rounded p-2 text-white" />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="text-sm text-purple-400 block mb-1">Learning Rate (α)</label>
                            <input type="number" step="0.1" min="0" max="1" value={alpha} onChange={e => setAlpha(parseFloat(e.target.value))} className="w-full bg-slate-900 border border-purple-900/50 rounded p-2 text-white" />
                        </div>
                        <div>
                            <label className="text-sm text-orange-400 block mb-1">Discount (γ)</label>
                            <input type="number" step="0.1" min="0" max="1" value={gamma} onChange={e => setGamma(parseFloat(e.target.value))} className="w-full bg-slate-900 border border-orange-900/50 rounded p-2 text-white" />
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="space-y-6">
                    {/* Step 1: Target */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                        <h4 className="text-sm text-slate-400 uppercase tracking-wider mb-2">1. The Target (Reality + Promise)</h4>
                        <div className="font-mono text-xl text-white">
                            {reward} + {gamma} × {qNextMax} = <span className="text-blue-400 font-bold">{target.toFixed(2)}</span>
                        </div>
                        <p className="text-xs text-slate-500 mt-2">Reward + Discounted Future Value</p>
                    </div>

                    {/* Step 2: Error */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
                        <h4 className="text-sm text-slate-400 uppercase tracking-wider mb-2">2. The Surprise (TD Error)</h4>
                        <div className="font-mono text-xl text-white">
                            {target.toFixed(2)} - {qOld} = <span className="text-red-400 font-bold">{tdError.toFixed(2)}</span>
                        </div>
                        <p className="text-xs text-slate-500 mt-2">Target - Old Expectation</p>
                    </div>

                    {/* Step 3: Update */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-green-500"></div>
                        <h4 className="text-sm text-slate-400 uppercase tracking-wider mb-2">3. The New Value</h4>
                        <div className="font-mono text-xl text-white">
                            {qOld} + {alpha} × {tdError.toFixed(2)} = <span className="text-green-400 font-bold text-3xl">{qNew.toFixed(2)}</span>
                        </div>
                        <p className="text-xs text-slate-500 mt-2">Nudging the old value towards the target</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
