import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const INITIAL_DATA = [
    { id: 'A', x: 10, y: 5 },
    { id: 'B', x: 50, y: 40 },
    { id: 'C', x: 30, y: 20 },
    { id: 'D', x: 20, y: 10 },
    { id: 'E', x: 40, y: 60 } // E has rank mismatch (X=4, Y=5)
];

const STEPS = [
    { id: 'raw', title: '1. Raw Data', desc: 'Start with raw (X, Y) pairs.' },
    { id: 'sortX', title: '2. Rank X', desc: 'Sort by X and assign Rank X (1 = smallest).' },
    { id: 'sortY', title: '3. Rank Y', desc: 'Sort by Y and assign Rank Y (1 = smallest).' },
    { id: 'diff', title: '4. Calculate d', desc: 'Difference d = Rank X - Rank Y' },
    { id: 'diffSq', title: '5. Square d', desc: 'Square the differences (d²)' },
    { id: 'sum', title: '6. Sum d²', desc: 'Sum all d² values.' },
    { id: 'formula', title: '7. Apply Formula', desc: 'ρ = 1 - (6 * Σd²) / (n(n² - 1))' }
];

export default function CalculationPanel() {
    const [step, setStep] = useState(0);

    // Helper to calculate ranks
    const getRankedData = () => {
        let data = [...INITIAL_DATA];

        // Rank X
        const sortedX = [...data].sort((a, b) => a.x - b.x);
        data = data.map(d => ({ ...d, rankX: sortedX.findIndex(i => i.id === d.id) + 1 }));

        // Rank Y
        const sortedY = [...data].sort((a, b) => a.y - b.y);
        data = data.map(d => ({ ...d, rankY: sortedY.findIndex(i => i.id === d.id) + 1 }));

        // Calcs
        data = data.map(d => ({
            ...d,
            d: d.rankX - d.rankY,
            d2: Math.pow(d.rankX - d.rankY, 2)
        }));

        return data;
    };

    const rankedData = getRankedData();
    const sumD2 = rankedData.reduce((sum, item) => sum + item.d2, 0);
    const n = rankedData.length;
    const rho = 1 - (6 * sumD2) / (n * (n * n - 1));

    // Determine sort order for display based on step
    let displayData = [...rankedData];
    if (step === 1) displayData.sort((a, b) => a.x - b.x);
    if (step === 2) displayData.sort((a, b) => a.y - b.y);
    if (step >= 3) displayData.sort((a, b) => a.id.localeCompare(b.id)); // Back to ID order for calc

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="w-full max-w-4xl mb-6 flex justify-between items-center">
                <h2 className="text-2xl font-bold text-slate-800">Step-by-Step Calculation</h2>
                <div className="flex gap-2">
                    <button
                        onClick={() => setStep(Math.max(0, step - 1))}
                        className="px-4 py-2 bg-slate-200 rounded hover:bg-slate-300 font-bold"
                        disabled={step === 0}
                    >
                        Previous
                    </button>
                    <button
                        onClick={() => setStep(Math.min(STEPS.length - 1, step + 1))}
                        className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 font-bold"
                        disabled={step === STEPS.length - 1}
                    >
                        Next
                    </button>
                </div>
            </div>

            <div className="w-full max-w-4xl bg-indigo-50 p-4 rounded-lg border border-indigo-200 mb-6">
                <h3 className="font-bold text-indigo-900">{STEPS[step].title}</h3>
                <p className="text-indigo-800">{STEPS[step].desc}</p>
            </div>

            <div className="w-full max-w-4xl overflow-hidden rounded-xl shadow-lg border border-slate-200 bg-white">
                <table className="w-full text-left border-collapse">
                    <thead className="bg-slate-100 text-slate-600 uppercase text-sm font-bold">
                        <tr>
                            <th className="p-4">ID</th>
                            <th className="p-4">X</th>
                            <th className="p-4">Y</th>
                            <th className={`p-4 transition-colors ${step >= 1 ? 'text-blue-600' : 'text-slate-300'}`}>Rank X</th>
                            <th className={`p-4 transition-colors ${step >= 2 ? 'text-green-600' : 'text-slate-300'}`}>Rank Y</th>
                            <th className={`p-4 transition-colors ${step >= 3 ? 'text-purple-600' : 'text-slate-300'}`}>d</th>
                            <th className={`p-4 transition-colors ${step >= 4 ? 'text-red-600' : 'text-slate-300'}`}>d²</th>
                        </tr>
                    </thead>
                    <tbody>
                        <AnimatePresence>
                            {displayData.map((row) => (
                                <motion.tr
                                    key={row.id}
                                    layout
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="border-b border-slate-100 hover:bg-slate-50"
                                >
                                    <td className="p-4 font-bold text-slate-700">{row.id}</td>
                                    <td className="p-4 text-slate-600">{row.x}</td>
                                    <td className="p-4 text-slate-600">{row.y}</td>
                                    <td className="p-4 font-mono font-bold text-blue-600">
                                        {step >= 1 ? row.rankX : '-'}
                                    </td>
                                    <td className="p-4 font-mono font-bold text-green-600">
                                        {step >= 2 ? row.rankY : '-'}
                                    </td>
                                    <td className="p-4 font-mono font-bold text-purple-600">
                                        {step >= 3 ? row.d : '-'}
                                    </td>
                                    <td className="p-4 font-mono font-bold text-red-600">
                                        {step >= 4 ? row.d2 : '-'}
                                    </td>
                                </motion.tr>
                            ))}
                        </AnimatePresence>
                    </tbody>
                    {step >= 5 && (
                        <tfoot className="bg-slate-50 font-bold">
                            <tr>
                                <td colSpan="6" className="p-4 text-right text-slate-600">Sum (Σd²):</td>
                                <td className="p-4 text-red-600 text-xl">{sumD2}</td>
                            </tr>
                        </tfoot>
                    )}
                </table>
            </div>

            {step >= 6 && (
                <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border-2 border-indigo-100 text-center animate-fade-in">
                    <p className="text-slate-500 text-sm mb-2">Formula Application</p>
                    <div className="text-2xl font-mono text-slate-800">
                        ρ = 1 - <span className="text-red-600">6({sumD2})</span> / <span className="text-blue-600">{n}({n}² - 1)</span>
                    </div>
                    <div className="text-4xl font-bold text-indigo-600 mt-4">
                        ρ = {rho.toFixed(3)}
                    </div>
                </div>
            )}
        </div>
    );
}
