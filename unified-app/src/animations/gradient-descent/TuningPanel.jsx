import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';

export default function TuningPanel() {
    const [learningRate, setLearningRate] = useState(0.1);
    const [history, setHistory] = useState([]);
    const [currentX, setCurrentX] = useState(-4);
    const [isRunning, setIsRunning] = useState(false);

    // Function: y = x^2
    const f = (x) => x * x;
    const df = (x) => 2 * x;

    useEffect(() => {
        let interval;
        if (isRunning) {
            interval = setInterval(() => {
                setCurrentX(prevX => {
                    const slope = df(prevX);
                    const nextX = prevX - learningRate * slope;

                    // Stop if converged or exploded
                    if (Math.abs(slope) < 0.01 || Math.abs(nextX) > 10) {
                        setIsRunning(false);
                    }

                    setHistory(prev => [...prev, { x: prevX, y: f(prevX) }]);
                    return nextX;
                });
            }, 200);
        }
        return () => clearInterval(interval);
    }, [isRunning, learningRate]);

    const startSimulation = () => {
        setCurrentX(-4);
        setHistory([]);
        setIsRunning(true);
    };

    // Generate curve data
    const curveData = [];
    for (let i = -5; i <= 5; i += 0.1) {
        curveData.push({ x: i, y: f(i) });
    }

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">Learning Rate Playground</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    The <strong>Learning Rate</strong> controls the step size.
                    <br />
                    Too small = Slow. Too big = Overshoot. Find the Goldilocks zone!
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-6xl">
                <div className="flex-1 bg-white p-6 rounded-xl shadow-lg border border-slate-200">
                    <label className="block font-bold text-slate-700 mb-4">Learning Rate: {learningRate}</label>
                    <input
                        type="range" min="0.01" max="1.1" step="0.01"
                        value={learningRate}
                        onChange={(e) => setLearningRate(Number(e.target.value))}
                        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600 mb-8"
                    />

                    <div className="flex gap-2 mb-4">
                        <button onClick={() => setLearningRate(0.01)} className="flex-1 py-2 bg-slate-100 rounded hover:bg-slate-200 text-xs font-bold">🐌 Too Slow (0.01)</button>
                        <button onClick={() => setLearningRate(0.1)} className="flex-1 py-2 bg-green-100 rounded hover:bg-green-200 text-xs font-bold text-green-800">✅ Just Right (0.1)</button>
                        <button onClick={() => setLearningRate(1.05)} className="flex-1 py-2 bg-red-100 rounded hover:bg-red-200 text-xs font-bold text-red-800">💥 Too Fast (1.05)</button>
                    </div>

                    <button
                        onClick={startSimulation}
                        disabled={isRunning}
                        className="w-full py-4 bg-indigo-600 text-white rounded-xl font-bold text-xl hover:bg-indigo-700 shadow-lg disabled:opacity-50"
                    >
                        {isRunning ? 'Running...' : '🚀 Run Simulation'}
                    </button>
                </div>

                <div className="flex-[2] h-[400px] bg-white rounded-xl shadow-lg border border-slate-200 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" type="number" domain={[-5, 5]} allowDataOverflow />
                            <YAxis dataKey="y" domain={[0, 25]} allowDataOverflow />
                            <Tooltip />
                            <Line data={curveData} type="monotone" dataKey="y" stroke="#cbd5e1" strokeWidth={2} dot={false} />

                            {/* History Path */}
                            {history.map((pt, i) => (
                                <ReferenceDot key={i} x={pt.x} y={pt.y} r={4} fill="#94a3b8" />
                            ))}

                            {/* Current Position */}
                            <ReferenceDot x={currentX} y={f(currentX)} r={8} fill={Math.abs(currentX) > 5 ? '#ef4444' : '#4f46e5'} stroke="white" strokeWidth={2} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
