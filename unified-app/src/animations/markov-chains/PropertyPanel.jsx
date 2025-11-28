import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function PropertyPanel() {
    const [currentPad, setCurrentPad] = useState(0); // 0, 1, 2
    const [history, setHistory] = useState([0]);
    const [isAuto, setIsAuto] = useState(false);

    // Transition probabilities: P(Next | Current)
    // Row i, Col j = P(j | i)
    const transitions = [
        [0.1, 0.6, 0.3], // From Pad 0
        [0.4, 0.2, 0.4], // From Pad 1
        [0.5, 0.3, 0.2]  // From Pad 2
    ];

    const pads = [
        { id: 0, label: 'Pad A', color: 'bg-emerald-500', x: '20%' },
        { id: 1, label: 'Pad B', color: 'bg-cyan-500', x: '50%' },
        { id: 2, label: 'Pad C', color: 'bg-indigo-500', x: '80%' }
    ];

    const jump = () => {
        const probs = transitions[currentPad];
        const rand = Math.random();
        let cumulative = 0;
        let nextPad = 0;

        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (rand < cumulative) {
                nextPad = i;
                break;
            }
        }

        setCurrentPad(nextPad);
        setHistory(prev => [...prev.slice(-9), nextPad]); // Keep last 10
    };

    useEffect(() => {
        let interval;
        if (isAuto) {
            interval = setInterval(jump, 1000);
        }
        return () => clearInterval(interval);
    }, [isAuto, currentPad]);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-400 mb-4">The Markov Property</h2>
                <p className="text-lg text-slate-300 leading-relaxed mb-4">
                    "The future depends <strong>only</strong> on the present, not the past."
                    <br />
                    <span className="text-sm text-slate-400">Memorylessness: P(X<sub>t+1</sub> | X<sub>t</sub>, X<sub>t-1</sub>...) = P(X<sub>t+1</sub> | X<sub>t</sub>)</span>
                </p>
            </div>

            {/* Pond Visualization */}
            <div className="relative w-full max-w-4xl h-64 bg-slate-800 rounded-xl border border-slate-700 mb-8 overflow-hidden">
                {/* Water */}
                <div className="absolute inset-0 bg-blue-900/20"></div>

                {/* Pads */}
                {pads.map((pad) => (
                    <div
                        key={pad.id}
                        className={`absolute top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-24 h-24 rounded-full border-4 border-slate-600 flex items-center justify-center ${pad.id === currentPad ? 'ring-4 ring-white/50' : ''}`}
                        style={{ left: pad.x }}
                    >
                        <div className={`w-20 h-20 rounded-full opacity-50 ${pad.color}`}></div>
                        <span className="absolute font-bold text-white text-xl">{pad.label}</span>

                        {/* Probabilities from current */}
                        {pad.id === currentPad && (
                            <div className="absolute -bottom-16 w-48 text-center text-xs text-slate-300 bg-slate-900/80 p-2 rounded">
                                Next Jump Probs:
                                <br />
                                A: {(transitions[pad.id][0] * 100).toFixed(0)}% |
                                B: {(transitions[pad.id][1] * 100).toFixed(0)}% |
                                C: {(transitions[pad.id][2] * 100).toFixed(0)}%
                            </div>
                        )}
                    </div>
                ))}

                {/* Frog */}
                <motion.div
                    className="absolute top-1/2 w-12 h-12 bg-green-400 rounded-full border-2 border-white shadow-lg z-10 flex items-center justify-center text-2xl"
                    animate={{
                        left: pads[currentPad].x,
                        y: [0, -50, 0], // Jump arc
                    }}
                    transition={{
                        left: { type: "spring", stiffness: 60, damping: 15 },
                        y: { duration: 0.4, times: [0, 0.5, 1] }
                    }}
                    style={{ x: '-50%', marginTop: '-24px' }}
                >
                    🐸
                </motion.div>
            </div>

            {/* Controls */}
            <div className="flex gap-4 mb-8">
                <button
                    onClick={jump}
                    className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold transition-all shadow-lg active:scale-95"
                >
                    Jump Once
                </button>
                <button
                    onClick={() => setIsAuto(!isAuto)}
                    className={`px-8 py-3 rounded-xl font-bold transition-all shadow-lg ${isAuto ? 'bg-red-600 hover:bg-red-500' : 'bg-blue-600 hover:bg-blue-500'} text-white`}
                >
                    {isAuto ? 'Stop Auto-Jump' : 'Start Auto-Jump'}
                </button>
            </div>

            {/* History Log */}
            <div className="w-full max-w-4xl bg-slate-800 p-4 rounded-xl border border-slate-700">
                <h3 className="text-sm font-bold text-slate-400 mb-2 uppercase tracking-wider">Jump History (Last 10)</h3>
                <div className="flex gap-2 justify-center">
                    <AnimatePresence>
                        {history.map((padId, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, scale: 0.5 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-white ${pads[padId].color}`}
                            >
                                {pads[padId].label.charAt(4)}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
                <p className="text-center text-xs text-slate-500 mt-3">
                    Notice: To decide the <em>next</em> jump, the frog only looks at the <strong>current</strong> pad.
                    <br />
                    It doesn't care where it was 5 jumps ago!
                </p>
            </div>
        </div>
    );
}
