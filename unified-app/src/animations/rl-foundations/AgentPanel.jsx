import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react';

export default function AgentPanel() {
    // Grid State: 4x4
    // 0 = Empty, 1 = Hole (Penalty), 2 = Goal (Reward)
    const [grid] = useState([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 2]
    ]);

    const [agentPos, setAgentPos] = useState({ r: 0, c: 0 });
    const [log, setLog] = useState([]);
    const [gameOver, setGameOver] = useState(false);

    const move = (dr, dc, actionName) => {
        if (gameOver) return;

        const newR = Math.min(3, Math.max(0, agentPos.r + dr));
        const newC = Math.min(3, Math.max(0, agentPos.c + dc));

        // Check if wall hit (didn't move)
        if (newR === agentPos.r && newC === agentPos.c) {
            setLog(prev => [`Hit Wall! Stayed at (${newR}, ${newC})`, ...prev.slice(0, 4)]);
            return;
        }

        const cellType = grid[newR][newC];
        let reward = -1; // Living cost
        let status = 'Moved';

        if (cellType === 1) {
            reward = -10;
            status = 'Fell in Hole!';
            setGameOver(true);
        } else if (cellType === 2) {
            reward = +10;
            status = 'Reached Goal!';
            setGameOver(true);
        }

        setAgentPos({ r: newR, c: newC });
        setLog(prev => [`Action: ${actionName} -> State: (${newR}, ${newC}) -> Reward: ${reward} (${status})`, ...prev.slice(0, 4)]);
    };

    const reset = () => {
        setAgentPos({ r: 0, c: 0 });
        setGameOver(false);
        setLog([]);
    };

    // Keyboard controls
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowUp') move(-1, 0, 'UP');
            if (e.key === 'ArrowDown') move(1, 0, 'DOWN');
            if (e.key === 'ArrowLeft') move(0, -1, 'LEFT');
            if (e.key === 'ArrowRight') move(0, 1, 'RIGHT');
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [agentPos, gameOver]);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-4">The Agent-Environment Loop</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    The Agent observes the <strong>State (S<sub>t</sub>)</strong>, takes an <strong>Action (A<sub>t</sub>)</strong>,
                    and receives a <strong>Reward (R<sub>t+1</sub>)</strong> and <strong>Next State (S<sub>t+1</sub>)</strong>.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Grid Visualization */}
                <div className="flex flex-col items-center">
                    <div className="relative bg-slate-800 p-4 rounded-xl border-4 border-slate-700 shadow-2xl">
                        <div className="grid grid-cols-4 gap-2">
                            {grid.map((row, r) => (
                                row.map((cell, c) => (
                                    <div
                                        key={`${r}-${c}`}
                                        className={`w-16 h-16 rounded flex items-center justify-center text-2xl relative ${cell === 1 ? 'bg-slate-900' : cell === 2 ? 'bg-yellow-900/50 border-2 border-yellow-500' : 'bg-slate-700'
                                            }`}
                                    >
                                        {cell === 1 && '🕳️'}
                                        {cell === 2 && '🏆'}

                                        {/* Coordinates */}
                                        <span className="absolute top-1 left-1 text-[10px] text-slate-700 dark:text-slate-500 font-mono">
                                            {r},{c}
                                        </span>

                                        {/* Agent */}
                                        {agentPos.r === r && agentPos.c === c && (
                                            <motion.div
                                                layoutId="agent"
                                                className="absolute inset-0 flex items-center justify-center text-3xl z-10"
                                            >
                                                🤖
                                            </motion.div>
                                        )}
                                    </div>
                                ))
                            ))}
                        </div>

                        {gameOver && (
                            <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg backdrop-blur-sm">
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-white mb-2">
                                        {grid[agentPos.r][agentPos.c] === 2 ? '🎉 You Win!' : '💀 Game Over'}
                                    </div>
                                    <button onClick={reset} className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded font-bold">
                                        Reset
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="mt-6 grid grid-cols-3 gap-2">
                        <div></div>
                        <button onClick={() => move(-1, 0, 'UP')} className="p-3 bg-slate-700 rounded hover:bg-slate-600 text-white"><ArrowUp /></button>
                        <div></div>
                        <button onClick={() => move(0, -1, 'LEFT')} className="p-3 bg-slate-700 rounded hover:bg-slate-600 text-white"><ArrowLeft /></button>
                        <button onClick={() => move(1, 0, 'DOWN')} className="p-3 bg-slate-700 rounded hover:bg-slate-600 text-white"><ArrowDown /></button>
                        <button onClick={() => move(0, 1, 'RIGHT')} className="p-3 bg-slate-700 rounded hover:bg-slate-600 text-white"><ArrowRight /></button>
                    </div>
                    <p className="text-xs text-slate-700 dark:text-slate-500 mt-2">Use Arrow Keys or Buttons</p>
                </div>

                {/* Interaction Log */}
                <div className="w-full">
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-[400px] flex flex-col">
                        <h3 className="font-bold text-white mb-4 border-b border-slate-700 pb-2">Interaction Log</h3>
                        <div className="flex-1 overflow-y-auto space-y-3 font-mono text-sm">
                            {log.length === 0 && <div className="text-slate-700 dark:text-slate-500 italic">Waiting for action...</div>}
                            {log.map((entry, i) => (
                                <div key={i} className="bg-slate-900 p-3 rounded border-l-4 border-emerald-500">
                                    {entry}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="mt-6 bg-emerald-900/20 border border-emerald-500/50 p-4 rounded-xl">
                        <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-2">Key Concepts</h4>
                        <ul className="list-disc list-inside text-sm text-slate-700 dark:text-slate-300 space-y-1">
                            <li><strong>State (S)</strong>: Where you are (e.g., 0,0).</li>
                            <li><strong>Action (A)</strong>: What you do (e.g., UP).</li>
                            <li><strong>Reward (R)</strong>: Feedback (e.g., -1 per step, +10 for goal).</li>
                            <li><strong>Policy (π)</strong>: The strategy (mapping S → A).</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
}
