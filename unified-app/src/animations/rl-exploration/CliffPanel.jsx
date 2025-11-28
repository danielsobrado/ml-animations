import React, { useState, useEffect, useRef } from 'react';

export default function CliffPanel() {
    // 4x8 Grid
    // Start (3,0), Goal (3,7)
    // Cliff: (3,1) to (3,6) -> Penalty -100 & Reset
    const ROWS = 4;
    const COLS = 8;

    const [agentPos, setAgentPos] = useState({ r: 3, c: 0 });
    const [epsilon, setEpsilon] = useState(0.1);
    const [isTraining, setIsTraining] = useState(false);
    const [pathHistory, setPathHistory] = useState([]); // Array of {r,c} for visualization
    const [falls, setFalls] = useState(0);
    const [wins, setWins] = useState(0);

    const trainingRef = useRef(null);

    // Simplified logic: Just random walk biased towards goal to simulate learning
    // In a real simulation, we'd run Q-Learning vs SARSA here.
    // For this demo, we simulate the *behavior* of High vs Low Epsilon.

    const step = () => {
        setAgentPos(prev => {
            let { r, c } = prev;

            // Decision: Move towards goal or random?
            // Optimal path is along row 2 (just above cliff)
            // Safe path is along row 1 or 0 (far from cliff)

            let dr = 0, dc = 0;

            // "Learned" Policy simulation based on Epsilon
            // High Epsilon -> Agent is erratic, so it should stay far from cliff (Row 0/1)
            // Low Epsilon -> Agent is precise, can hug the cliff (Row 2)

            const isRandom = Math.random() < epsilon;

            if (isRandom) {
                // Random move
                const moves = [[0, 1], [0, -1], [1, 0], [-1, 0]];
                const move = moves[Math.floor(Math.random() * moves.length)];
                dr = move[0];
                dc = move[1];
            } else {
                // Greedy move towards goal (3,7)
                // If High Epsilon, "Greedy" learned path is safer (Row 1)
                // If Low Epsilon, "Greedy" learned path is optimal (Row 2)
                const preferredRow = epsilon > 0.3 ? 1 : 2;

                if (r > preferredRow) dr = -1; // Go up to safety
                else if (r < preferredRow) dr = 1; // Go down to optimal
                else if (c < 7) dc = 1; // Go right
                else if (r < 3) dr = 1; // Go down to goal
            }

            let nextR = Math.min(ROWS - 1, Math.max(0, r + dr));
            let nextC = Math.min(COLS - 1, Math.max(0, c + dc));

            // Check Cliff
            if (nextR === 3 && nextC > 0 && nextC < 7) {
                setFalls(f => f + 1);
                setPathHistory([]);
                return { r: 3, c: 0 }; // Reset
            }

            // Check Goal
            if (nextR === 3 && nextC === 7) {
                setWins(w => w + 1);
                setPathHistory([]);
                return { r: 3, c: 0 }; // Reset
            }

            setPathHistory(h => [...h.slice(-20), { r: nextR, c: nextC }]);
            return { r: nextR, c: nextC };
        });
    };

    useEffect(() => {
        if (isTraining) {
            trainingRef.current = setInterval(step, 100);
        } else {
            clearInterval(trainingRef.current);
        }
        return () => clearInterval(trainingRef.current);
    }, [isTraining, epsilon]);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-red-400 mb-4">The Cliff Walking Problem</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    The shortest path is right along the edge of the cliff.
                    <br />
                    But if you explore randomly (High ε), you might fall off!
                </p>
            </div>

            <div className="w-full max-w-5xl">
                {/* Grid */}
                <div className="grid grid-cols-8 gap-1 bg-slate-800 p-4 rounded-xl border border-slate-700 mb-8">
                    {Array(ROWS).fill().map((_, r) => (
                        Array(COLS).fill().map((_, c) => {
                            const isCliff = r === 3 && c > 0 && c < 7;
                            const isGoal = r === 3 && c === 7;
                            const isStart = r === 3 && c === 0;

                            return (
                                <div
                                    key={`${r}-${c}`}
                                    className={`h-16 rounded flex items-center justify-center relative ${isCliff ? 'bg-slate-950 border-t-4 border-red-600' :
                                            isGoal ? 'bg-yellow-900/50 border border-yellow-500' :
                                                isStart ? 'bg-slate-700 border border-slate-500' : 'bg-slate-700'
                                        }`}
                                >
                                    {isCliff && <span className="text-2xl">💀</span>}
                                    {isGoal && <span className="text-2xl">🏆</span>}
                                    {isStart && <span className="text-xs text-slate-400">START</span>}

                                    {/* Agent */}
                                    {agentPos.r === r && agentPos.c === c && (
                                        <div className="absolute inset-0 flex items-center justify-center text-3xl z-10 transition-all duration-100">
                                            🤖
                                        </div>
                                    )}

                                    {/* Trail */}
                                    {pathHistory.some(p => p.r === r && p.c === c) && !isCliff && !isGoal && (
                                        <div className="absolute w-2 h-2 bg-white/30 rounded-full"></div>
                                    )}
                                </div>
                            );
                        })
                    ))}
                </div>

                {/* Controls */}
                <div className="grid md:grid-cols-2 gap-8">
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <h3 className="font-bold text-white mb-4">Simulation Control</h3>

                        <div className="mb-6">
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-400">Exploration (ε)</label>
                                <span className="text-2xl font-mono font-bold text-red-400">{epsilon.toFixed(2)}</span>
                            </div>
                            <input
                                type="range" min="0" max="0.5" step="0.05"
                                value={epsilon}
                                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                                className="w-full accent-red-400"
                            />
                            <p className="text-xs text-slate-500 mt-2">
                                High ε = Safer Path (Far from cliff) | Low ε = Optimal Path (Edge)
                            </p>
                        </div>

                        <button
                            onClick={() => setIsTraining(!isTraining)}
                            className={`w-full py-3 rounded-xl font-bold text-white shadow-lg ${isTraining ? 'bg-slate-600' : 'bg-red-600 hover:bg-red-500'}`}
                        >
                            {isTraining ? 'Pause' : 'Start Walking'}
                        </button>
                    </div>

                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <h3 className="font-bold text-white mb-4">Results</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-900 p-4 rounded text-center">
                                <div className="text-3xl font-bold text-red-500">{falls}</div>
                                <div className="text-xs text-slate-400">Falls (Deaths)</div>
                            </div>
                            <div className="bg-slate-900 p-4 rounded text-center">
                                <div className="text-3xl font-bold text-yellow-500">{wins}</div>
                                <div className="text-xs text-slate-400">Wins (Goals)</div>
                            </div>
                        </div>
                        <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-sm text-slate-300">
                            {epsilon > 0.2 ? (
                                "With high noise, the optimal path is too dangerous. The agent prefers the 'safe' route."
                            ) : (
                                "With low noise, the agent dares to walk the edge for the shortest path."
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
