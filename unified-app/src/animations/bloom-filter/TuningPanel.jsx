import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';

export default function TuningPanel() {
    const [m, setM] = useState(100); // Bit array size
    const [n, setN] = useState(10);  // Expected items
    const [k, setK] = useState(3);   // Hash functions

    // Calculate False Positive Probability: P ≈ (1 - e^(-kn/m))^k
    const calculateP = (mVal, nVal, kVal) => {
        const exponent = -1 * kVal * nVal / mVal;
        const base = 1 - Math.exp(exponent);
        return Math.pow(base, kVal);
    };

    const currentP = calculateP(m, n, k);
    const optimalK = (m / n) * Math.log(2);

    // Generate data for the chart (varying k)
    const data = [];
    for (let i = 1; i <= 15; i++) {
        data.push({
            k: i,
            p: calculateP(m, n, i),
            isCurrent: i === k
        });
    }

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Tuning Studio</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Find the sweet spot! Adjust the parameters to minimize the <strong>False Positive Probability</strong>.
                </p>
            </div>

            <div className="flex flex-col lg:flex-row gap-8 w-full max-w-6xl">
                {/* Controls */}
                <div className="flex-1 bg-white p-6 rounded-xl shadow-lg border border-slate-200">
                    <div className="mb-6">
                        <label className="block font-bold text-slate-700 mb-2">Bit Array Size (m): {m}</label>
                        <input
                            type="range" min="10" max="1000" step="10" value={m}
                            onChange={(e) => setM(Number(e.target.value))}
                            className="w-full accent-indigo-600"
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block font-bold text-slate-700 mb-2">Expected Items (n): {n}</label>
                        <input
                            type="range" min="1" max="100" step="1" value={n}
                            onChange={(e) => setN(Number(e.target.value))}
                            className="w-full accent-indigo-600"
                        />
                    </div>
                    <div className="mb-6">
                        <label className="block font-bold text-slate-700 mb-2">Hash Functions (k): {k}</label>
                        <input
                            type="range" min="1" max="15" step="1" value={k}
                            onChange={(e) => setK(Number(e.target.value))}
                            className="w-full accent-indigo-600"
                        />
                    </div>

                    <div className="mt-8 p-4 bg-slate-50 rounded-lg border border-slate-200">
                        <h4 className="font-bold text-slate-500 uppercase text-xs mb-2">Optimal k for current m, n</h4>
                        <p className="text-2xl font-mono font-bold text-slate-800">{optimalK.toFixed(1)}</p>
                        <p className="text-xs text-slate-400 mt-1">Formula: k = (m/n) * ln(2)</p>
                    </div>
                </div>

                {/* Visualization */}
                <div className="flex-[2] flex flex-col">
                    <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 h-[400px] mb-6">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="k" label={{ value: 'Number of Hash Functions (k)', position: 'bottom', offset: 0 }} />
                                <YAxis label={{ value: 'False Positive Probability', angle: -90, position: 'left' }} />
                                <Tooltip
                                    formatter={(val) => [(val * 100).toFixed(2) + '%', 'Probability']}
                                    labelFormatter={(label) => `k = ${label}`}
                                />
                                <Line type="monotone" dataKey="p" stroke="#4f46e5" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                                <ReferenceDot x={k} y={currentP} r={8} fill="#f59e0b" stroke="white" strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className={`p-6 rounded-xl border-2 text-center transition-colors ${currentP < 0.01 ? 'bg-green-50 border-green-200 text-green-900' : (currentP < 0.1 ? 'bg-yellow-50 border-yellow-200 text-yellow-900' : 'bg-red-50 border-red-200 text-red-900')}`}>
                        <p className="text-sm font-bold uppercase opacity-70 mb-1">Current Probability</p>
                        <p className="text-4xl font-mono font-bold">{(currentP * 100).toFixed(4)}%</p>
                        <p className="text-sm mt-2">
                            {currentP < 0.01 ? "✅ Excellent! Very rare collisions." : (currentP < 0.1 ? "⚠️ Acceptable for some use cases." : "🚨 Too high! Increase size (m) or adjust (k).")}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
