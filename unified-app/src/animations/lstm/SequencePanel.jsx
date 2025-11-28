import React, { useState } from 'react';

export default function SequencePanel() {
    const [timeStep, setTimeStep] = useState(1);

    return (
        <div className="p-8 h-full flex flex-col items-center justify-center">
            <h2 className="text-2xl font-bold text-slate-800 mb-8">Sequence View (Unfolding in Time)</h2>

            <div className="flex items-center gap-4 overflow-x-auto p-8 w-full justify-center">
                {/* Time Step t-1 */}
                <div className={`transition-all duration-500 ${timeStep >= 0 ? 'opacity-100' : 'opacity-30'}`}>
                    <div className="w-32 h-48 bg-slate-200 rounded-xl border-4 border-slate-300 flex items-center justify-center relative">
                        <span className="font-bold text-slate-400 text-xl">LSTM<br />t-1</span>
                        {/* Arrows */}
                        <div className="absolute -right-8 top-1/2 w-8 h-1 bg-slate-400"></div>
                    </div>
                </div>

                {/* Time Step t */}
                <div className={`transition-all duration-500 transform ${timeStep >= 1 ? 'scale-110 border-indigo-500 shadow-xl' : 'scale-100'}`}>
                    <div className={`w-32 h-48 bg-white rounded-xl border-4 ${timeStep >= 1 ? 'border-indigo-500' : 'border-slate-300'} flex items-center justify-center relative`}>
                        <span className={`font-bold text-xl ${timeStep >= 1 ? 'text-indigo-600' : 'text-slate-400'}`}>LSTM<br />t</span>
                        {/* Arrows */}
                        <div className="absolute -right-8 top-1/2 w-8 h-1 bg-slate-400"></div>
                    </div>
                </div>

                {/* Time Step t+1 */}
                <div className={`transition-all duration-500 ${timeStep >= 2 ? 'opacity-100' : 'opacity-30'}`}>
                    <div className="w-32 h-48 bg-slate-200 rounded-xl border-4 border-slate-300 flex items-center justify-center">
                        <span className="font-bold text-slate-400 text-xl">LSTM<br />t+1</span>
                    </div>
                </div>
            </div>

            <div className="mt-8 flex gap-4">
                <button
                    onClick={() => setTimeStep(Math.max(0, timeStep - 1))}
                    className="px-6 py-2 bg-slate-200 rounded-lg font-bold hover:bg-slate-300"
                >
                    Previous
                </button>
                <button
                    onClick={() => setTimeStep(Math.min(2, timeStep + 1))}
                    className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700"
                >
                    Next Time Step
                </button>
            </div>

            <div className="mt-6 text-center text-slate-600 max-w-md">
                <p>
                    The <strong>Hidden State (h)</strong> and <strong>Cell State (C)</strong> are passed from one time step to the next. This is how the network remembers context!
                </p>
            </div>
        </div>
    );
}
