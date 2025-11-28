import React, { useState } from 'react';

export default function RewardPanel() {
    // 3x3 Grid for simplicity in design mode
    const [grid, setGrid] = useState([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 2] // Goal at bottom right
    ]);

    const [selectedTool, setSelectedTool] = useState(1); // 0=Empty, 1=Hole, 2=Goal, 3=Gold

    const tools = [
        { id: 0, label: 'Empty (-1)', icon: '⬜', reward: -1, color: 'bg-slate-700' },
        { id: 1, label: 'Hole (-10)', icon: '🕳️', reward: -10, color: 'bg-slate-900' },
        { id: 2, label: 'Goal (+50)', icon: '🏆', reward: 50, color: 'bg-yellow-600' },
        { id: 3, label: 'Gold (+5)', icon: '💰', reward: 5, color: 'bg-amber-500' }
    ];

    const updateCell = (r, c) => {
        const newGrid = [...grid];
        newGrid[r] = [...newGrid[r]];
        newGrid[r][c] = selectedTool;
        setGrid(newGrid);
    };

    // Calculate a hypothetical path score (simple heuristic)
    // Assume path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
    const path = [
        { r: 0, c: 0 }, { r: 0, c: 1 }, { r: 0, c: 2 }, { r: 1, c: 2 }, { r: 2, c: 2 }
    ];

    const calculatePathReturn = () => {
        let total = 0;
        path.forEach((pos, i) => {
            if (i === 0) return; // Start doesn't give reward immediately usually, but let's simplify
            const type = grid[pos.r][pos.c];
            const tool = tools.find(t => t.id === type);
            total += tool.reward;
        });
        return total;
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-amber-400 mb-4">Designing Rewards</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    The <strong>Reward Function</strong> defines the goal.
                    <br />
                    The agent will do <em>anything</em> to maximize the total reward. Be careful what you wish for!
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl">
                {/* Editor */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-4 text-center">Level Editor</h3>

                    {/* Toolbar */}
                    <div className="flex justify-center gap-2 mb-6">
                        {tools.map(tool => (
                            <button
                                key={tool.id}
                                onClick={() => setSelectedTool(tool.id)}
                                className={`flex flex-col items-center p-2 rounded-lg transition-all ${selectedTool === tool.id
                                        ? 'bg-slate-600 ring-2 ring-amber-400 scale-105'
                                        : 'bg-slate-700 hover:bg-slate-600'
                                    }`}
                            >
                                <span className="text-2xl">{tool.icon}</span>
                                <span className="text-xs text-slate-300 mt-1">{tool.label}</span>
                            </button>
                        ))}
                    </div>

                    {/* Grid */}
                    <div className="flex justify-center">
                        <div className="grid grid-cols-3 gap-2 bg-slate-900 p-2 rounded-lg">
                            {grid.map((row, r) => (
                                row.map((cell, c) => (
                                    <button
                                        key={`${r}-${c}`}
                                        onClick={() => updateCell(r, c)}
                                        className={`w-20 h-20 rounded flex items-center justify-center text-3xl transition-all hover:brightness-110 ${tools.find(t => t.id === cell).color
                                            }`}
                                    >
                                        {tools.find(t => t.id === cell).icon}
                                        {/* Show path overlay */}
                                        {path.some(p => p.r === r && p.c === c) && (
                                            <div className="absolute w-2 h-2 bg-white rounded-full opacity-50"></div>
                                        )}
                                    </button>
                                ))
                            ))}
                        </div>
                    </div>
                    <p className="text-center text-xs text-slate-500 mt-2">Click grid to place items</p>
                </div>

                {/* Analysis */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-4">Reward Analysis</h3>

                    <div className="mb-6">
                        <h4 className="text-sm text-slate-400 mb-2">Hypothetical Path (White Dots)</h4>
                        <div className="bg-slate-900 p-4 rounded-lg font-mono text-sm">
                            {path.map((pos, i) => {
                                if (i === 0) return null;
                                const type = grid[pos.r][pos.c];
                                const tool = tools.find(t => t.id === type);
                                return (
                                    <div key={i} className="flex justify-between border-b border-slate-800 py-1 last:border-0">
                                        <span>Step {i}: ({pos.r},{pos.c}) {tool.icon}</span>
                                        <span className={tool.reward > 0 ? 'text-green-400' : 'text-red-400'}>
                                            {tool.reward > 0 ? '+' : ''}{tool.reward}
                                        </span>
                                    </div>
                                );
                            })}
                            <div className="flex justify-between border-t border-slate-700 pt-2 mt-2 font-bold text-lg">
                                <span>Total Return:</span>
                                <span className={calculatePathReturn() > 0 ? 'text-green-400' : 'text-red-400'}>
                                    {calculatePathReturn()}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-amber-900/20 border border-amber-500/50 p-4 rounded-xl">
                        <h4 className="font-bold text-amber-400 mb-2">Reward Hacking ⚠️</h4>
                        <p className="text-sm text-slate-300">
                            If you put too much <strong>Gold 💰</strong> everywhere, the agent might just run around collecting gold forever instead of reaching the <strong>Goal 🏆</strong>!
                            <br /><br />
                            This is called <em>Reward Hacking</em>. The agent optimizes for the number, not your intent.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
