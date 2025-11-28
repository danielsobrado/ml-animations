import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Simple non-crypto hash function simulation
const hash1 = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
    return Math.abs(hash);
};

const hash2 = (str) => {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) hash = (hash * 33) ^ str.charCodeAt(i);
    return Math.abs(hash);
};

const hash3 = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) hash = (hash + str.charCodeAt(i) * (i + 1));
    return Math.abs(hash);
};

const BIT_ARRAY_SIZE = 20;

export default function PlaygroundPanel() {
    const [bits, setBits] = useState(Array(BIT_ARRAY_SIZE).fill(0));
    const [input, setInput] = useState('');
    const [lastAction, setLastAction] = useState(null); // { type: 'add' | 'check', word: string, indices: number[], result: boolean }

    const getIndices = (word) => {
        return [
            hash1(word) % BIT_ARRAY_SIZE,
            hash2(word) % BIT_ARRAY_SIZE,
            hash3(word) % BIT_ARRAY_SIZE
        ];
    };

    const handleAdd = () => {
        if (!input) return;
        const indices = getIndices(input);
        const newBits = [...bits];
        indices.forEach(idx => newBits[idx] = 1);
        setBits(newBits);
        setLastAction({ type: 'add', word: input, indices, result: true });
        setInput('');
    };

    const handleCheck = () => {
        if (!input) return;
        const indices = getIndices(input);
        const allSet = indices.every(idx => bits[idx] === 1);
        setLastAction({ type: 'check', word: input, indices, result: allSet });
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Interactive Playground</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Type a word to <strong>Add</strong> it (flip bits to 1) or <strong>Check</strong> if it exists.
                </p>
            </div>

            {/* Input Area */}
            <div className="flex gap-4 mb-12 w-full max-w-md">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type a word..."
                    className="flex-1 px-4 py-3 rounded-lg border-2 border-slate-300 focus:border-indigo-500 outline-none font-bold text-lg"
                    onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                />
                <button
                    onClick={handleAdd}
                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 transition-colors"
                >
                    Add
                </button>
                <button
                    onClick={handleCheck}
                    className="px-6 py-3 bg-teal-600 text-white rounded-lg font-bold hover:bg-teal-700 transition-colors"
                >
                    Check
                </button>
            </div>

            {/* Bit Array Visualization */}
            <div className="w-full max-w-5xl bg-slate-100 p-8 rounded-2xl border border-slate-200 mb-8 relative">
                <div className="flex justify-between mb-2 text-xs font-mono text-slate-400 px-1">
                    {bits.map((_, i) => <span key={i}>{i}</span>)}
                </div>
                <div className="flex gap-1 h-16">
                    {bits.map((bit, i) => {
                        const isHighlighted = lastAction?.indices.includes(i);
                        return (
                            <motion.div
                                key={i}
                                initial={false}
                                animate={{
                                    backgroundColor: bit === 1 ? '#4f46e5' : '#e2e8f0',
                                    scale: isHighlighted ? 1.1 : 1,
                                    borderColor: isHighlighted ? (lastAction?.type === 'add' ? '#f59e0b' : '#14b8a6') : 'transparent'
                                }}
                                className={`flex-1 rounded border-2 flex items-center justify-center font-bold text-xl transition-colors ${bit === 1 ? 'text-white' : 'text-slate-300'}`}
                            >
                                {bit}
                            </motion.div>
                        );
                    })}
                </div>

                {/* Connection Lines (Simplified visual representation) */}
                <AnimatePresence>
                    {lastAction && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className="absolute top-full left-0 w-full text-center mt-4"
                        >
                            <div className={`inline-block px-6 py-3 rounded-lg font-bold text-lg shadow-lg ${lastAction.type === 'add'
                                    ? 'bg-indigo-100 text-indigo-900'
                                    : (lastAction.result ? 'bg-green-100 text-green-900' : 'bg-red-100 text-red-900')
                                }`}>
                                {lastAction.type === 'add' && `Added "${lastAction.word}" → Flipped bits ${lastAction.indices.join(', ')}`}
                                {lastAction.type === 'check' && (
                                    lastAction.result
                                        ? `"${lastAction.word}" is PROBABLY in the set (All bits are 1)`
                                        : `"${lastAction.word}" is DEFINITELY NOT in the set (Found a 0)`
                                )}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full">
                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                    <h3 className="font-bold text-indigo-900 mb-2">Why "Probably"?</h3>
                    <p className="text-slate-600 text-sm">
                        If all bits are 1, it <em>might</em> be because we added this word... or it might be a coincidence from other words flipping those same bits! This is a <strong>False Positive</strong>.
                    </p>
                </div>
                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                    <h3 className="font-bold text-indigo-900 mb-2">Why "Definitely Not"?</h3>
                    <p className="text-slate-600 text-sm">
                        If even a single bit is 0, we know for sure we never added this word. Because if we had, that bit would have been flipped to 1! No <strong>False Negatives</strong> allowed.
                    </p>
                </div>
            </div>
        </div>
    );
}
