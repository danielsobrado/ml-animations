import React, { useState } from 'react';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react';

export default function TablePanel() {
    // 3x3 Grid
    // State 8 (2,2) is Goal
    const [qTable, setQTable] = useState(
        Array(3).fill().map(() =>
            Array(3).fill().map(() => ({ up: 0, down: 0, left: 0, right: 0 }))
        )
    );

    const [hoveredCell, setHoveredCell] = useState(null);

    // Helper to color based on value
    const getColor = (val) => {
        if (val > 0) return `rgba(34, 197, 94, ${Math.min(1, val / 10)})`; // Green
        if (val < 0) return `rgba(239, 68, 68, ${Math.min(1, Math.abs(val) / 10)})`; // Red
        return 'rgba(148, 163, 184, 0.1)'; // Grey
    };

    const updateQ = (r, c, action, value) => {
        const newTable = [...qTable];
        newTable[r] = [...newTable[r]];
        newTable[r][c] = { ...newTable[r][c], [action]: parseFloat(value) };
        setQTable(newTable);
    };

    // Pre-fill with a learned policy
    const loadLearned = () => {
        const newTable = JSON.parse(JSON.stringify(qTable));
        // Path to goal (2,2)
        newTable[0][0].right = 5; newTable[0][0].down = 5;
        newTable[0][1].right = 6;
        newTable[0][2].down = 7;
        newTable[1][2].down = 8;
        newTable[2][2] = { up: 0, down: 0, left: 0, right: 0 }; // Goal
        setQTable(newTable);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-cyan-400 mb-4">The Q-Table</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    The agent's brain is a table of values: <strong>Q(State, Action)</strong>.
                    <br />
                    "If I am in State S and take Action A, how much total reward do I expect?"
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Grid Visualization */}
                <div className="flex flex-col items-center">
                    <div className="grid grid-cols-3 gap-4 bg-slate-800 p-4 rounded-xl border border-slate-700">
                        {qTable.map((row, r) => (
                            row.map((cell, c) => (
                                <div
                                    key={`${r}-${c}`}
                                    onMouseEnter={() => setHoveredCell({ r, c })}
                                    onMouseLeave={() => setHoveredCell(null)}
                                    className={`relative w-24 h-24 bg-slate-900 rounded-lg border-2 ${r === 2 && c === 2 ? 'border-yellow-500' : 'border-slate-700'
                                        } flex items-center justify-center`}
                                >
                                    {/* Triangles for Q-values */}
                                    {/* UP */}
                                    <div
                                        className="absolute top-0 left-0 right-0 h-8 flex justify-center pt-1"
                                        style={{ backgroundColor: getColor(cell.up), clipPath: 'polygon(50% 100%, 0 0, 100% 0)' }}
                                    >
                                        <ArrowUp size={12} className="text-white/50" />
                                    </div>
                                    {/* DOWN */}
                                    <div
                                        className="absolute bottom-0 left-0 right-0 h-8 flex justify-center items-end pb-1"
                                        style={{ backgroundColor: getColor(cell.down), clipPath: 'polygon(50% 0, 0 100%, 100% 100%)' }}
                                    >
                                        <ArrowDown size={12} className="text-white/50" />
                                    </div>
                                    {/* LEFT */}
                                    <div
                                        className="absolute top-0 bottom-0 left-0 w-8 flex items-center pl-1"
                                        style={{ backgroundColor: getColor(cell.left), clipPath: 'polygon(100% 50%, 0 0, 0 100%)' }}
                                    >
                                        <ArrowLeft size={12} className="text-white/50" />
                                    </div>
                                    {/* RIGHT */}
                                    <div
                                        className="absolute top-0 bottom-0 right-0 w-8 flex items-center justify-end pr-1"
                                        style={{ backgroundColor: getColor(cell.right), clipPath: 'polygon(0 50%, 100% 0, 100% 100%)' }}
                                    >
                                        <ArrowRight size={12} className="text-white/50" />
                                    </div>

                                    {/* Center Label */}
                                    <span className="z-10 font-mono text-xs text-slate-400 pointer-events-none">
                                        {r},{c}
                                    </span>
                                    {r === 2 && c === 2 && <span className="absolute text-2xl z-20">🏆</span>}
                                </div>
                            ))
                        ))}
                    </div>
                    <button
                        onClick={loadLearned}
                        className="mt-6 px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-bold shadow-lg"
                    >
                        Load Learned Policy
                    </button>
                </div>

                {/* Inspector */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 w-full">
                    <h3 className="font-bold text-white mb-4 border-b border-slate-700 pb-2">
                        Inspector: State ({hoveredCell ? hoveredCell.r : '-'}, {hoveredCell ? hoveredCell.c : '-'})
                    </h3>

                    {hoveredCell ? (
                        <div className="space-y-4">
                            {['up', 'down', 'left', 'right'].map(action => (
                                <div key={action} className="flex items-center gap-4">
                                    <span className="w-16 font-bold text-slate-300 uppercase">{action}</span>
                                    <input
                                        type="number"
                                        value={qTable[hoveredCell.r][hoveredCell.c][action]}
                                        onChange={(e) => updateQ(hoveredCell.r, hoveredCell.c, action, e.target.value)}
                                        className="bg-slate-900 border border-slate-600 rounded px-3 py-2 text-white w-24"
                                    />
                                    <div className="flex-1 h-2 bg-slate-900 rounded overflow-hidden">
                                        <div
                                            className="h-full transition-all"
                                            style={{
                                                width: `${Math.min(100, Math.abs(qTable[hoveredCell.r][hoveredCell.c][action]) * 10)}%`,
                                                backgroundColor: qTable[hoveredCell.r][hoveredCell.c][action] >= 0 ? '#22c55e' : '#ef4444'
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                            <p className="text-xs text-slate-500 mt-4">
                                The agent chooses the action with the <strong>highest Q-value</strong> (Greedy Policy).
                            </p>
                        </div>
                    ) : (
                        <div className="text-slate-500 italic text-center py-12">
                            Hover over a grid cell to inspect and edit Q-values.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
