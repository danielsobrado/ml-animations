import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot, ReferenceLine } from 'recharts';

export default function SlopePanel() {
    const [x, setX] = useState(-4); // Start position
    const [history, setHistory] = useState([-4]);
    const learningRate = 0.1;

    // Function: y = x^2
    const f = (val) => val * val;
    const df = (val) => 2 * val; // Derivative

    const data = [];
    for (let i = -5; i <= 5; i += 0.5) {
        data.push({ x: i, y: f(i) });
    }

    const slope = df(x);
    const stepSize = slope * learningRate;
    const nextX = x - stepSize;

    const takeStep = () => {
        setX(nextX);
        setHistory([...history, nextX]);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">2D Slope Lab</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Connecting the hiker to the math. The "Slope" is the <strong>Derivative</strong>.
                    <br />
                    Update Rule: <span className="font-mono bg-slate-100 p-1 rounded">New Position = Old Position - (Learning Rate × Slope)</span>
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl">
                {/* Chart */}
                <div className="flex-1 h-[400px] bg-white rounded-xl shadow-lg border border-slate-200 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" type="number" domain={[-5, 5]} allowDataOverflow />
                            <YAxis dataKey="y" />
                            <Tooltip />
                            <Line type="monotone" dataKey="y" stroke="#4f46e5" strokeWidth={3} dot={false} />

                            {/* Current Position */}
                            <ReferenceDot x={x} y={f(x)} r={8} fill="#ef4444" stroke="white" strokeWidth={2} />

                            {/* Tangent Line (Visual approximation) */}
                            <ReferenceLine
                                segment={[
                                    { x: x - 1, y: f(x) - slope },
                                    { x: x + 1, y: f(x) + slope }
                                ]}
                                stroke="#22c55e"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Math Panel */}
                <div className="flex-1 flex flex-col justify-center gap-6">
                    <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                        <h3 className="font-bold text-slate-500 uppercase text-xs mb-4">Current State</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <p className="text-sm text-slate-500">Position (x)</p>
                                <p className="text-2xl font-mono font-bold text-slate-800">{x.toFixed(3)}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-500">Slope (dy/dx)</p>
                                <p className={`text-2xl font-mono font-bold ${slope > 0 ? 'text-orange-600' : 'text-blue-600'}`}>
                                    {slope.toFixed(3)}
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-indigo-50 p-6 rounded-xl border border-indigo-200">
                        <h3 className="font-bold text-indigo-900 uppercase text-xs mb-4">Calculation</h3>
                        <div className="font-mono text-sm space-y-2">
                            <p>Step = Learning Rate × Slope</p>
                            <p className="pl-4">= {learningRate} × {slope.toFixed(3)}</p>
                            <p className="pl-4 font-bold text-indigo-600">= {stepSize.toFixed(3)}</p>
                            <div className="h-px bg-indigo-200 my-2"></div>
                            <p>New x = {x.toFixed(3)} - {stepSize.toFixed(3)}</p>
                            <p className="font-bold text-xl text-indigo-700">= {nextX.toFixed(3)}</p>
                        </div>
                    </div>

                    <button
                        onClick={takeStep}
                        className="w-full py-4 bg-indigo-600 text-white rounded-xl font-bold text-xl hover:bg-indigo-700 shadow-lg transition-all transform hover:scale-105"
                    >
                        📉 Descend Step
                    </button>

                    <button
                        onClick={() => { setX(-4); setHistory([-4]); }}
                        className="w-full py-2 bg-slate-200 text-slate-700 rounded-lg font-bold hover:bg-slate-300"
                    >
                        Reset
                    </button>
                </div>
            </div>
        </div>
    );
}
