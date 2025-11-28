import React, { useState, useRef } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function InteractivePanel() {
    const [points, setPoints] = useState([]);
    const svgRef = useRef(null);

    // Calculate OLS
    const calculateOLS = () => {
        if (points.length < 2) return null;

        const n = points.length;
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

        points.forEach(p => {
            sumX += p.x;
            sumY += p.y;
            sumXY += p.x * p.y;
            sumXX += p.x * p.x;
        });

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return { slope, intercept };
    };

    const model = calculateOLS();

    const handleMouseDown = (e) => {
        if (!svgRef.current) return;
        // Simple coordinate mapping (approximate for demo, ideally use scale functions)
        // For this demo, we'll just add random points near click or use a dedicated "Add Point" button approach
        // to avoid complex coordinate math without D3 scales.
        // Instead, let's use a "Click to Add" interaction area that maps 0-100% to graph domain.

        const rect = svgRef.current.getBoundingClientRect();
        const xPct = (e.clientX - rect.left) / rect.width;
        const yPct = 1 - (e.clientY - rect.top) / rect.height; // Invert Y

        const x = xPct * 10; // Domain 0-10
        const y = yPct * 10; // Domain 0-10

        setPoints([...points, { x, y }]);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Interactive Fitter</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Click anywhere to add data points.
                    <br />
                    The <strong>Least Squares Line</strong> updates instantly. Watch how outliers pull the line!
                </p>
            </div>

            <div className="w-full max-w-4xl h-[500px] bg-white rounded-xl shadow-lg border border-slate-200 p-4 relative cursor-crosshair">
                {/* Click Area Overlay */}
                <div
                    ref={svgRef}
                    className="absolute inset-0 z-10"
                    onClick={handleMouseDown}
                ></div>

                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={[0, 10]} />
                        <YAxis dataKey="y" type="number" domain={[0, 10]} />

                        {/* The Best Fit Line */}
                        {model && (
                            <ReferenceLine
                                segment={[{ x: 0, y: model.intercept }, { x: 10, y: model.slope * 10 + model.intercept }]}
                                stroke="#4f46e5"
                                strokeWidth={4}
                            />
                        )}

                        {/* Data Points */}
                        <Scatter data={points} fill="#ef4444" />
                    </ScatterChart>
                </ResponsiveContainer>

                {/* Stats Overlay */}
                <div className="absolute top-4 right-4 bg-white/90 p-4 rounded-lg shadow border border-slate-200 pointer-events-none z-20">
                    <h4 className="font-bold text-slate-500 text-xs uppercase mb-2">Equation</h4>
                    {model ? (
                        <div className="font-mono text-lg font-bold text-indigo-700">
                            y = {model.slope.toFixed(2)}x + {model.intercept.toFixed(2)}
                        </div>
                    ) : (
                        <div className="text-slate-400 italic">Add at least 2 points</div>
                    )}
                    <div className="mt-2 text-xs text-slate-500">n = {points.length}</div>
                </div>

                <button
                    onClick={() => setPoints([])}
                    className="absolute bottom-4 right-4 px-4 py-2 bg-white text-red-600 border border-red-200 rounded-lg shadow-sm hover:bg-red-50 font-bold text-sm z-20"
                >
                    Clear All
                </button>
            </div>
        </div>
    );
}
