import React, { useState } from 'react';

export default function BuilderPanel() {
    // 3 States: Sunny (0), Cloudy (1), Rainy (2)
    // Matrix P[i][j] = P(j | i) (From i to j)
    const [matrix, setMatrix] = useState([
        [0.7, 0.2, 0.1], // Sunny -> S, C, R
        [0.3, 0.4, 0.3], // Cloudy -> S, C, R
        [0.2, 0.3, 0.5]  // Rainy -> S, C, R
    ]);

    const states = ['☀️ Sunny', '☁️ Cloudy', '🌧️ Rainy'];
    const colors = ['text-yellow-400', 'text-slate-400', 'text-blue-400'];
    const bgColors = ['bg-yellow-900/30', 'bg-slate-800', 'bg-blue-900/30'];

    const updateMatrix = (row, col, value) => {
        const newVal = Math.min(1, Math.max(0, parseFloat(value) || 0));
        const newMatrix = [...matrix];
        newMatrix[row] = [...newMatrix[row]];
        newMatrix[row][col] = newVal;
        setMatrix(newMatrix);
    };

    // Check row sums
    const rowSums = matrix.map(row => row.reduce((a, b) => a + b, 0));
    const isValid = rowSums.every(sum => Math.abs(sum - 1) < 0.01);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-blue-400 mb-4">Transition Matrix</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    The <strong>Transition Matrix (P)</strong> defines the rules of the world.
                    <br />
                    Row <em>i</em>, Column <em>j</em> = Probability of going from State <em>i</em> to State <em>j</em>.
                </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8 w-full max-w-6xl">
                {/* Matrix Editor */}
                <div className="bg-slate-800 p-6 rounded-xl border border-blue-500/50">
                    <h3 className="font-bold text-white mb-6 text-center text-xl">Matrix P</h3>

                    <div className="grid grid-cols-[auto_1fr_1fr_1fr] gap-4 items-center mb-4">
                        {/* Header */}
                        <div></div>
                        {states.map((s, i) => (
                            <div key={i} className={`text-center font-bold ${colors[i]}`}>To {s.split(' ')[1]}</div>
                        ))}

                        {/* Rows */}
                        {states.map((fromState, i) => (
                            <React.Fragment key={i}>
                                <div className={`font-bold ${colors[i]} text-right pr-2`}>From {fromState.split(' ')[1]}</div>
                                {matrix[i].map((val, j) => (
                                    <div key={`${i}-${j}`} className="relative">
                                        <input
                                            type="number"
                                            step="0.1"
                                            min="0"
                                            max="1"
                                            value={val}
                                            onChange={(e) => updateMatrix(i, j, e.target.value)}
                                            className={`w-full bg-slate-900 border ${rowSums[i] > 1.01 || rowSums[i] < 0.99 ? 'border-red-500' : 'border-slate-600'} rounded p-2 text-center text-white focus:border-blue-500 outline-none`}
                                        />
                                    </div>
                                ))}
                            </React.Fragment>
                        ))}
                    </div>

                    {!isValid && (
                        <div className="mt-4 p-3 bg-red-900/30 rounded-lg border border-red-700 text-center text-red-300 text-sm">
                            ⚠️ Warning: Probabilities in each row must sum to 1.0!
                            <br />
                            Current sums: {rowSums.map(s => s.toFixed(2)).join(', ')}
                        </div>
                    )}
                </div>

                {/* Graph Visualization (Simplified) */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 relative h-[400px]">
                    <h3 className="font-bold text-white mb-4 text-center text-xl">State Graph</h3>

                    {/* Nodes positioned in a triangle */}
                    <div className="absolute top-10 left-1/2 transform -translate-x-1/2 text-center">
                        <div className={`w-24 h-24 rounded-full border-4 border-yellow-500 flex items-center justify-center text-4xl bg-slate-900 z-10 relative`}>☀️</div>
                        <div className="text-yellow-400 font-bold mt-2">Sunny</div>
                    </div>

                    <div className="absolute bottom-10 left-10 text-center">
                        <div className={`w-24 h-24 rounded-full border-4 border-slate-500 flex items-center justify-center text-4xl bg-slate-900 z-10 relative`}>☁️</div>
                        <div className="text-slate-400 font-bold mt-2">Cloudy</div>
                    </div>

                    <div className="absolute bottom-10 right-10 text-center">
                        <div className={`w-24 h-24 rounded-full border-4 border-blue-500 flex items-center justify-center text-4xl bg-slate-900 z-10 relative`}>🌧️</div>
                        <div className="text-blue-400 font-bold mt-2">Rainy</div>
                    </div>

                    {/* Arrows (SVG Overlay) */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                        <defs>
                            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
                            </marker>
                        </defs>
                        {/* Simplified connections for visualization - dynamic lines would require complex geometry math */}
                        <path d="M 280 140 L 150 280" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" opacity={matrix[0][1] > 0 ? 1 : 0.1} />
                        <text x="200" y="200" fill="#cbd5e1" fontSize="12">{matrix[0][1]}</text>

                        <path d="M 180 280 L 380 280" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" opacity={matrix[1][2] > 0 ? 1 : 0.1} />
                        <text x="280" y="270" fill="#cbd5e1" fontSize="12">{matrix[1][2]}</text>

                        <path d="M 400 280 L 300 140" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" opacity={matrix[2][0] > 0 ? 1 : 0.1} />
                        <text x="360" y="200" fill="#cbd5e1" fontSize="12">{matrix[2][0]}</text>

                        {/* Self loops */}
                        <circle cx="300" cy="80" r="20" fill="none" stroke="#64748b" strokeWidth="2" opacity={matrix[0][0] > 0 ? 1 : 0.1} />
                        <text x="300" y="50" fill="#cbd5e1" fontSize="12" textAnchor="middle">{matrix[0][0]}</text>
                    </svg>

                    <div className="absolute bottom-2 right-2 text-xs text-slate-500">
                        * Simplified graph view (showing main cycle)
                    </div>
                </div>
            </div>
        </div>
    );
}
