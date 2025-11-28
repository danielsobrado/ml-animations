import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// Data: Monotonic but non-linear (e.g., y = x^3)
const RAW_DATA = [
    { id: 1, x: 1, y: 1, rankX: 1, rankY: 1 },
    { id: 2, x: 2, y: 8, rankX: 2, rankY: 2 },
    { id: 3, x: 3, y: 27, rankX: 3, rankY: 3 },
    { id: 4, x: 4, y: 64, rankX: 4, rankY: 4 },
    { id: 5, x: 5, y: 125, rankX: 5, rankY: 5 },
    { id: 6, x: 6, y: 216, rankX: 6, rankY: 6 },
    { id: 7, x: 10, y: 1000, rankX: 7, rankY: 7 }, // Outlier in value, but just next in rank
];

export default function ConceptPanel() {
    const [mode, setMode] = useState('raw'); // 'raw' or 'rank'

    const data = RAW_DATA.map(d => ({
        ...d,
        xVal: mode === 'raw' ? d.x : d.rankX,
        yVal: mode === 'raw' ? d.y : d.rankY,
    }));

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Power of Ranks</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Spearman correlation doesn't care about the <strong>values</strong> (how much).
                    It only cares about the <strong>ranks</strong> (which is larger).
                </p>
            </div>

            <div className="flex gap-4 mb-8">
                <button
                    onClick={() => setMode('raw')}
                    className={`px-6 py-3 rounded-lg font-bold transition-colors ${mode === 'raw' ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-600'}`}
                >
                    Show Raw Values
                </button>
                <button
                    onClick={() => setMode('rank')}
                    className={`px-6 py-3 rounded-lg font-bold transition-colors ${mode === 'rank' ? 'bg-green-600 text-white' : 'bg-slate-200 text-slate-600'}`}
                >
                    Show Ranks
                </button>
            </div>

            <div className="w-full max-w-4xl h-[400px] bg-slate-50 rounded-xl border border-slate-200 p-4 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid />
                        <XAxis
                            type="number"
                            dataKey="xVal"
                            name={mode === 'raw' ? "Value X" : "Rank X"}
                            domain={mode === 'raw' ? [0, 12] : [0, 8]}
                            label={{ value: mode === 'raw' ? "Raw Value X" : "Rank X", position: 'bottom', offset: 0 }}
                        />
                        <YAxis
                            type="number"
                            dataKey="yVal"
                            name={mode === 'raw' ? "Value Y" : "Rank Y"}
                            domain={mode === 'raw' ? [0, 1100] : [0, 8]}
                            label={{ value: mode === 'raw' ? "Raw Value Y" : "Rank Y", angle: -90, position: 'left' }}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Data" data={data} fill="#8884d8">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={mode === 'raw' ? '#3b82f6' : '#22c55e'} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>

                <div className={`absolute top-4 right-4 p-4 rounded-lg shadow-lg ${mode === 'raw' ? 'bg-blue-100 text-blue-900' : 'bg-green-100 text-green-900'}`}>
                    <p className="font-bold">{mode === 'raw' ? 'Non-Linear Curve' : 'Perfect Straight Line'}</p>
                    <p className="text-sm">
                        {mode === 'raw'
                            ? 'Pearson correlation is low because it expects a straight line.'
                            : 'Spearman correlation is 1.0 because the ranks match perfectly!'}
                    </p>
                </div>
            </div>
        </div>
    );
}
