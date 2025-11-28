import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// Initial linear data
const INITIAL_POINTS = [
    { id: 1, x: 10, y: 10 },
    { id: 2, x: 20, y: 20 },
    { id: 3, x: 30, y: 30 },
    { id: 4, x: 40, y: 40 },
    { id: 5, x: 50, y: 50 },
    { id: 6, x: 60, y: 60 },
    { id: 7, x: 70, y: 70 },
];

export default function RobustnessPanel() {
    const [points, setPoints] = useState(INITIAL_POINTS);
    const [outlierVal, setOutlierVal] = useState(80); // Y value of the last point

    // Update the last point based on slider
    const currentPoints = points.map(p =>
        p.id === 7 ? { ...p, y: outlierVal } : p
    );

    // Calculate Pearson (Simplified)
    const calcPearson = (data) => {
        const n = data.length;
        const sumX = data.reduce((a, b) => a + b.x, 0);
        const sumY = data.reduce((a, b) => a + b.y, 0);
        const sumXY = data.reduce((a, b) => a + b.x * b.y, 0);
        const sumX2 = data.reduce((a, b) => a + b.x * b.x, 0);
        const sumY2 = data.reduce((a, b) => a + b.y * b.y, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        return numerator / denominator;
    };

    // Calculate Spearman
    const calcSpearman = (data) => {
        const sortedX = [...data].sort((a, b) => a.x - b.x);
        const sortedY = [...data].sort((a, b) => a.y - b.y);

        const ranked = data.map(d => ({
            ...d,
            rankX: sortedX.findIndex(i => i.id === d.id) + 1,
            rankY: sortedY.findIndex(i => i.id === d.id) + 1
        }));

        const n = data.length;
        const sumD2 = ranked.reduce((sum, d) => sum + Math.pow(d.rankX - d.rankY, 2), 0);
        return 1 - (6 * sumD2) / (n * (n * n - 1));
    };

    const pearson = calcPearson(currentPoints);
    const spearman = calcSpearman(currentPoints);

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Robustness to Outliers</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Drag the slider to create an extreme outlier. Watch how <strong>Pearson</strong> panics, while <strong>Spearman</strong> stays calm.
                </p>
            </div>

            <div className="w-full max-w-md bg-white p-6 rounded-xl shadow-lg border border-slate-200 mb-8">
                <label className="block text-sm font-bold text-slate-700 mb-2">Outlier Y Value: {outlierVal}</label>
                <input
                    type="range"
                    min="0"
                    max="1000"
                    step="10"
                    value={outlierVal}
                    onChange={(e) => setOutlierVal(Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>0</span>
                    <span>500</span>
                    <span>1000 (Extreme!)</span>
                </div>
            </div>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-5xl">
                {/* Chart */}
                <div className="flex-1 h-[300px] bg-slate-50 rounded-xl border border-slate-200 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid />
                            <XAxis type="number" dataKey="x" name="X" domain={[0, 80]} />
                            <YAxis type="number" dataKey="y" name="Y" domain={[0, 1000]} />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Scatter name="Data" data={currentPoints} fill="#8884d8">
                                {currentPoints.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={index === 6 ? '#ef4444' : '#3b82f6'} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>

                {/* Metrics */}
                <div className="flex-1 flex flex-col gap-4 justify-center">
                    <div className="bg-blue-50 p-6 rounded-xl border-2 border-blue-200 transition-all duration-300">
                        <h3 className="text-blue-900 font-bold uppercase tracking-wider text-sm mb-1">Pearson Correlation</h3>
                        <p className="text-4xl font-mono font-bold text-blue-600">{pearson.toFixed(3)}</p>
                        <p className="text-blue-800 text-sm mt-2">
                            {pearson < 0.5 ? "😱 Ruined by the outlier!" : "✅ Strong linear relationship"}
                        </p>
                    </div>

                    <div className="bg-green-50 p-6 rounded-xl border-2 border-green-200 transition-all duration-300">
                        <h3 className="text-green-900 font-bold uppercase tracking-wider text-sm mb-1">Spearman Correlation</h3>
                        <p className="text-4xl font-mono font-bold text-green-600">{spearman.toFixed(3)}</p>
                        <p className="text-green-800 text-sm mt-2">
                            {spearman > 0.9 ? "😎 Still perfect! (Rank didn't change)" : "🤔 Rank changed significantly"}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
