import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

export default function ResidualsPanel() {
    const [slope, setSlope] = useState(1);
    const [intercept, setIntercept] = useState(0);
    const [showSquares, setShowSquares] = useState(true);

    // Fixed dataset
    const data = [
        { x: 1, y: 2 },
        { x: 2, y: 3 },
        { x: 3, y: 5 },
        { x: 4, y: 4 },
        { x: 5, y: 6 }
    ];

    // Calculate residuals and total error (MSE)
    let totalSquaredError = 0;
    const residuals = data.map(point => {
        const predictedY = slope * point.x + intercept;
        const error = point.y - predictedY;
        totalSquaredError += error * error;
        return { ...point, predictedY, error };
    });

    const mse = totalSquaredError / data.length;

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Residuals</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    The goal is to minimize the <strong>Mean Squared Error (MSE)</strong>.
                    <br />
                    Drag the sliders to adjust the line. Can you make the squares smaller?
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl">
                {/* Chart Area */}
                <div className="flex-1 h-[500px] bg-white rounded-xl shadow-lg border border-slate-200 p-4 relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" type="number" domain={[0, 6]} />
                            <YAxis dataKey="y" type="number" domain={[0, 8]} />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />

                            {/* The Line */}
                            <ReferenceLine
                                segment={[{ x: 0, y: intercept }, { x: 6, y: slope * 6 + intercept }]}
                                stroke="#4f46e5"
                                strokeWidth={3}
                            />

                            {/* Residual Lines (Vertical) */}
                            {residuals.map((point, i) => (
                                <ReferenceLine
                                    key={`res-${i}`}
                                    segment={[{ x: point.x, y: point.y }, { x: point.x, y: point.predictedY }]}
                                    stroke="#ef4444"
                                    strokeDasharray="3 3"
                                />
                            ))}

                            {/* Data Points */}
                            <Scatter data={data} fill="#0f172a">
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>

                    {/* Visual Squares (Overlay using HTML/CSS for simplicity over SVG complexity in Recharts) */}
                    {showSquares && (
                        <div className="absolute inset-0 pointer-events-none overflow-hidden">
                            {/* Note: Accurate pixel mapping would require D3 or custom SVG. 
                   For this Recharts implementation, we visualize the concept via the error metric below 
                   and simple vertical lines above. A full "square" visualization is complex to overlay 
                   perfectly on a responsive Recharts container without access to its internal scale.
                   We will rely on the "Total Error" number and the vertical red lines for feedback.
               */}
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="w-full md:w-80 flex flex-col gap-6">
                    <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                        <h3 className="font-bold text-slate-500 uppercase text-xs mb-4">Model Parameters</h3>

                        <div className="mb-6">
                            <label className="flex justify-between text-sm font-bold text-slate-700 mb-2">
                                Slope (m): <span className="text-indigo-600">{slope.toFixed(2)}</span>
                            </label>
                            <input
                                type="range" min="-2" max="4" step="0.1"
                                value={slope}
                                onChange={(e) => setSlope(Number(e.target.value))}
                                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                            />
                        </div>

                        <div className="mb-6">
                            <label className="flex justify-between text-sm font-bold text-slate-700 mb-2">
                                Intercept (b): <span className="text-indigo-600">{intercept.toFixed(2)}</span>
                            </label>
                            <input
                                type="range" min="-2" max="8" step="0.1"
                                value={intercept}
                                onChange={(e) => setIntercept(Number(e.target.value))}
                                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                            />
                        </div>
                    </div>

                    <div className={`p-6 rounded-xl border-2 transition-colors ${mse < 0.5 ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                        <h3 className="font-bold uppercase text-xs mb-2 opacity-70">Mean Squared Error</h3>
                        <p className="text-4xl font-mono font-bold text-slate-800">{mse.toFixed(3)}</p>
                        <p className="text-xs mt-2 text-slate-500">Lower is better!</p>
                    </div>

                    <button
                        onClick={() => { setSlope(1); setIntercept(1); }}
                        className="py-3 bg-slate-200 text-slate-700 rounded-xl font-bold hover:bg-slate-300 transition-colors"
                    >
                        Reset to Default
                    </button>
                </div>
            </div>
        </div>
    );
}
