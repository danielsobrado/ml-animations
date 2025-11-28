import React, { useState, useEffect } from 'react';
import { Package, Trash2, PlusSquare, Camera } from 'lucide-react';

export default function ConceptPanel() {
    const [packages, setPackages] = useState([]);
    const [step, setStep] = useState(0);

    // Simulation of the conveyor belt
    useEffect(() => {
        const interval = setInterval(() => {
            setPackages(prev => {
                const newPackages = prev.map(p => ({ ...p, x: p.x + 1 })).filter(p => p.x < 100);
                if (Math.random() > 0.7 && newPackages.length < 5) {
                    newPackages.push({ id: Date.now(), x: 0, content: '📦', status: 'fresh' });
                }
                return newPackages;
            });
        }, 100);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Conveyor Belt Analogy</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Imagine an LSTM as a factory worker standing next to a <strong>conveyor belt</strong>.
                    The belt carries packages (Information) from the past into the future.
                    The worker can do three things:
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl mb-12">
                <div className="bg-red-50 p-6 rounded-xl border-2 border-red-100 hover:border-red-300 transition-colors">
                    <div className="bg-red-100 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto text-red-600">
                        <Trash2 size={24} />
                    </div>
                    <h3 className="text-xl font-bold text-red-900 text-center mb-2">1. Forget</h3>
                    <p className="text-red-800 text-center text-sm">
                        "This package is old/irrelevant." <br />
                        <strong>The Forget Gate</strong> removes info from the cell state.
                    </p>
                </div>

                <div className="bg-green-50 p-6 rounded-xl border-2 border-green-100 hover:border-green-300 transition-colors">
                    <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto text-green-600">
                        <PlusSquare size={24} />
                    </div>
                    <h3 className="text-xl font-bold text-green-900 text-center mb-2">2. Add</h3>
                    <p className="text-green-800 text-center text-sm">
                        "Here's something new to remember." <br />
                        <strong>The Input Gate</strong> adds new info to the cell state.
                    </p>
                </div>

                <div className="bg-blue-50 p-6 rounded-xl border-2 border-blue-100 hover:border-blue-300 transition-colors">
                    <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mb-4 mx-auto text-blue-600">
                        <Camera size={24} />
                    </div>
                    <h3 className="text-xl font-bold text-blue-900 text-center mb-2">3. Output</h3>
                    <p className="text-blue-800 text-center text-sm">
                        "Report current status." <br />
                        <strong>The Output Gate</strong> reads the state without changing it.
                    </p>
                </div>
            </div>

            {/* Visual Animation Area */}
            <div className="w-full max-w-4xl bg-slate-100 rounded-2xl p-8 relative overflow-hidden h-64 border-b-4 border-slate-300">
                <div className="absolute top-0 left-0 w-full h-full flex items-center">
                    {/* Conveyor Belt Track */}
                    <div className="w-full h-4 bg-slate-300 relative">
                        <div className="absolute top-0 left-0 h-full w-full animate-slide bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTAgMGg0MHYxNkgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0wIDBoMnYxNmgtMnoiIGZpbGw9IiM5NDc0ODgiIGZpbGwtb3BhY2l0eT0iLjIiLz48L3N2Zz4=')]"></div>
                    </div>
                </div>

                {/* Packages */}
                {packages.map(p => (
                    <div
                        key={p.id}
                        className="absolute top-1/2 -translate-y-1/2 transition-all duration-100 text-4xl"
                        style={{ left: `${p.x}%` }}
                    >
                        {p.content}
                    </div>
                ))}

                {/* Worker Stations */}
                <div className="absolute top-1/2 -translate-y-1/2 left-[30%] flex flex-col items-center">
                    <div className="w-1 h-16 bg-red-400 mb-2"></div>
                    <span className="bg-red-100 text-red-800 text-xs font-bold px-2 py-1 rounded">Forget Gate</span>
                </div>

                <div className="absolute top-1/2 -translate-y-1/2 left-[50%] flex flex-col items-center">
                    <div className="w-1 h-16 bg-green-400 mb-2"></div>
                    <span className="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded">Input Gate</span>
                </div>

                <div className="absolute top-1/2 -translate-y-1/2 left-[70%] flex flex-col items-center">
                    <div className="w-1 h-16 bg-blue-400 mb-2"></div>
                    <span className="bg-blue-100 text-blue-800 text-xs font-bold px-2 py-1 rounded">Output Gate</span>
                </div>
            </div>

            <div className="mt-8 bg-yellow-50 p-4 rounded-lg border border-yellow-200 max-w-2xl">
                <p className="text-yellow-900 text-center font-medium">
                    <strong>Key Insight:</strong> The conveyor belt (Cell State) runs straight through the entire chain with only minor linear interactions. This allows information to flow unchanged for a long time!
                </p>
            </div>
        </div>
    );
}
