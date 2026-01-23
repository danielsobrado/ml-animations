import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Layers, Plus, Info, Zap, TrendingUp } from 'lucide-react';

// Simple 2D vector visualization
function Vector2D({ x, y, color, label, maxMagnitude = 3 }) {
    const scale = 60; // pixels per unit
    const clampedX = Math.max(-maxMagnitude, Math.min(maxMagnitude, x));
    const clampedY = Math.max(-maxMagnitude, Math.min(maxMagnitude, y));

    return (
        <g>
            {/* Arrow line */}
            <motion.line
                x1={0}
                y1={0}
                x2={clampedX * scale}
                y2={-clampedY * scale}
                stroke={color}
                strokeWidth={3}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.5 }}
            />
            {/* Arrow head */}
            <motion.circle
                cx={clampedX * scale}
                cy={-clampedY * scale}
                r={6}
                fill={color}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
            />
            {/* Label */}
            <text
                x={clampedX * scale + 10}
                y={-clampedY * scale}
                fill={color}
                fontSize={12}
                fontWeight="bold"
            >
                {label}
            </text>
        </g>
    );
}

function LayerBlock({ layerNum, output, onOutputChange, color }) {
    return (
        <div className={`p-4 rounded-xl border-2 ${color}`}>
            <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-slate-700 dark:text-slate-200">
                    Layer {layerNum} Output
                </span>
                <span className="text-xs text-slate-500 font-mono">
                    f{layerNum}(x)
                </span>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-xs text-slate-500 mb-1">X component</label>
                    <input
                        type="range"
                        min="-1"
                        max="1"
                        step="0.1"
                        value={output[0]}
                        onChange={(e) => onOutputChange([parseFloat(e.target.value), output[1]])}
                        className="w-full"
                    />
                    <div className="text-center font-mono text-sm">{output[0].toFixed(1)}</div>
                </div>
                <div>
                    <label className="block text-xs text-slate-500 mb-1">Y component</label>
                    <input
                        type="range"
                        min="-1"
                        max="1"
                        step="0.1"
                        value={output[1]}
                        onChange={(e) => onOutputChange([output[0], parseFloat(e.target.value)])}
                        className="w-full"
                    />
                    <div className="text-center font-mono text-sm">{output[1].toFixed(1)}</div>
                </div>
            </div>
        </div>
    );
}

export default function PlaygroundPanel() {
    const [numLayers, setNumLayers] = useState(3);
    const [inputVector, setInputVector] = useState([1.0, 0.5]);
    const [layerOutputs, setLayerOutputs] = useState([
        [0.3, 0.4],
        [-0.2, 0.5],
        [0.4, -0.3],
        [0.1, 0.2],
        [-0.3, 0.3],
    ]);

    // Compute residual stream at each layer
    const computeResiduals = useCallback(() => {
        const residuals = [inputVector];
        for (let i = 0; i < numLayers; i++) {
            const prev = residuals[i];
            const layerOut = layerOutputs[i];
            residuals.push([prev[0] + layerOut[0], prev[1] + layerOut[1]]);
        }
        return residuals;
    }, [inputVector, layerOutputs, numLayers]);

    const residuals = computeResiduals();

    const updateLayerOutput = (layerIdx, newOutput) => {
        const newOutputs = [...layerOutputs];
        newOutputs[layerIdx] = newOutput;
        setLayerOutputs(newOutputs);
    };

    const randomize = () => {
        setInputVector([
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2,
        ]);
        setLayerOutputs(
            layerOutputs.map(() => [
                (Math.random() - 0.5) * 1.5,
                (Math.random() - 0.5) * 1.5,
            ])
        );
    };

    const reset = () => {
        setInputVector([1.0, 0.5]);
        setLayerOutputs([
            [0.3, 0.4],
            [-0.2, 0.5],
            [0.4, -0.3],
            [0.1, 0.2],
            [-0.3, 0.3],
        ]);
    };

    const colors = [
        '#06b6d4', // cyan - input
        '#8b5cf6', // violet - layer 1
        '#f59e0b', // amber - layer 2
        '#10b981', // emerald - layer 3
        '#f43f5e', // rose - layer 4
        '#6366f1', // indigo - layer 5
    ];

    const layerColors = [
        'border-violet-300 dark:border-violet-700 bg-violet-50 dark:bg-violet-900/20',
        'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20',
        'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20',
        'border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-900/20',
        'border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20',
    ];

    const finalMagnitude = Math.sqrt(
        residuals[numLayers][0] ** 2 + residuals[numLayers][1] ** 2
    );

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-6xl w-full">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold text-indigo-600 dark:text-indigo-400 mb-4">
                        Interactive Residual Stream
                    </h2>
                    <p className="text-lg text-slate-700 dark:text-slate-300">
                        Adjust layer outputs and watch how they accumulate in the residual stream
                    </p>
                </div>

                {/* Controls */}
                <div className="flex flex-wrap justify-center gap-4 mb-8">
                    <div className="flex items-center gap-2 bg-white dark:bg-slate-800 px-4 py-2 rounded-lg border border-slate-200 dark:border-slate-700">
                        <label className="text-sm text-slate-600 dark:text-slate-400">Layers:</label>
                        <select
                            value={numLayers}
                            onChange={(e) => setNumLayers(parseInt(e.target.value))}
                            className="bg-transparent text-slate-700 dark:text-slate-200 font-medium"
                        >
                            {[1, 2, 3, 4, 5].map((n) => (
                                <option key={n} value={n}>{n}</option>
                            ))}
                        </select>
                    </div>
                    <button
                        onClick={randomize}
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
                    >
                        <RefreshCw size={18} />
                        Randomize
                    </button>
                    <button
                        onClick={reset}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                    >
                        Reset
                    </button>
                </div>

                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Vector Visualization */}
                    <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                        <h3 className="text-lg font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                            <Zap className="text-indigo-500" size={20} />
                            Vector Space Visualization
                        </h3>

                        <div className="flex justify-center">
                            <svg width="350" height="350" className="bg-slate-50 dark:bg-slate-900 rounded-xl">
                                {/* Grid */}
                                <g transform="translate(175, 175)">
                                    {/* Axes */}
                                    <line x1="-150" y1="0" x2="150" y2="0" stroke="#94a3b8" strokeWidth="1" />
                                    <line x1="0" y1="-150" x2="0" y2="150" stroke="#94a3b8" strokeWidth="1" />

                                    {/* Grid lines */}
                                    {[-2, -1, 1, 2].map((v) => (
                                        <React.Fragment key={v}>
                                            <line
                                                x1={v * 60} y1="-150" x2={v * 60} y2="150"
                                                stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4"
                                            />
                                            <line
                                                x1="-150" y1={-v * 60} x2="150" y2={-v * 60}
                                                stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4"
                                            />
                                        </React.Fragment>
                                    ))}

                                    {/* Show all layer contribution vectors from origin */}
                                    {Array.from({ length: numLayers }).map((_, i) => (
                                        <Vector2D
                                            key={`layer-${i}`}
                                            x={layerOutputs[i][0]}
                                            y={layerOutputs[i][1]}
                                            color={colors[i + 1]}
                                            label={`f${i + 1}`}
                                        />
                                    ))}

                                    {/* Input vector */}
                                    <Vector2D
                                        x={inputVector[0]}
                                        y={inputVector[1]}
                                        color={colors[0]}
                                        label="Input"
                                    />

                                    {/* Final residual (sum of all) */}
                                    <Vector2D
                                        x={residuals[numLayers][0]}
                                        y={residuals[numLayers][1]}
                                        color="#22c55e"
                                        label="Final"
                                    />
                                </g>
                            </svg>
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colors[0] }} />
                                <span className="text-slate-600 dark:text-slate-400">Input: [{inputVector[0].toFixed(2)}, {inputVector[1].toFixed(2)}]</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-green-500" />
                                <span className="text-slate-600 dark:text-slate-400">
                                    Final: [{residuals[numLayers][0].toFixed(2)}, {residuals[numLayers][1].toFixed(2)}]
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Layer Controls */}
                    <div className="space-y-4">
                        {/* Input Vector Control */}
                        <div className="p-4 rounded-xl border-2 border-cyan-300 dark:border-cyan-700 bg-cyan-50 dark:bg-cyan-900/20">
                            <div className="flex items-center justify-between mb-3">
                                <span className="font-semibold text-slate-700 dark:text-slate-200">
                                    Input Vector
                                </span>
                                <span className="text-xs text-slate-500 font-mono">x</span>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">X component</label>
                                    <input
                                        type="range"
                                        min="-2"
                                        max="2"
                                        step="0.1"
                                        value={inputVector[0]}
                                        onChange={(e) => setInputVector([parseFloat(e.target.value), inputVector[1]])}
                                        className="w-full"
                                    />
                                    <div className="text-center font-mono text-sm">{inputVector[0].toFixed(1)}</div>
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 mb-1">Y component</label>
                                    <input
                                        type="range"
                                        min="-2"
                                        max="2"
                                        step="0.1"
                                        value={inputVector[1]}
                                        onChange={(e) => setInputVector([inputVector[0], parseFloat(e.target.value)])}
                                        className="w-full"
                                    />
                                    <div className="text-center font-mono text-sm">{inputVector[1].toFixed(1)}</div>
                                </div>
                            </div>
                        </div>

                        {/* Layer outputs */}
                        {Array.from({ length: numLayers }).map((_, i) => (
                            <LayerBlock
                                key={i}
                                layerNum={i + 1}
                                output={layerOutputs[i]}
                                onOutputChange={(newOutput) => updateLayerOutput(i, newOutput)}
                                color={layerColors[i]}
                            />
                        ))}
                    </div>
                </div>

                {/* Residual Stream Breakdown */}
                <div className="mt-8 bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
                    <h3 className="text-lg font-bold text-slate-700 dark:text-slate-200 mb-4 flex items-center gap-2">
                        <TrendingUp className="text-green-500" size={20} />
                        Residual Stream at Each Layer
                    </h3>

                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-200 dark:border-slate-700">
                                    <th className="text-left py-2 px-4 text-slate-600 dark:text-slate-400">Position</th>
                                    <th className="text-left py-2 px-4 text-slate-600 dark:text-slate-400">Formula</th>
                                    <th className="text-left py-2 px-4 text-slate-600 dark:text-slate-400">Value</th>
                                    <th className="text-left py-2 px-4 text-slate-600 dark:text-slate-400">Magnitude</th>
                                </tr>
                            </thead>
                            <tbody>
                                {residuals.map((vec, i) => {
                                    const mag = Math.sqrt(vec[0] ** 2 + vec[1] ** 2);
                                    return (
                                        <motion.tr
                                            key={i}
                                            className="border-b border-slate-100 dark:border-slate-700/50"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.1 }}
                                        >
                                            <td className="py-2 px-4">
                                                <div className="flex items-center gap-2">
                                                    <div
                                                        className="w-3 h-3 rounded-full"
                                                        style={{ backgroundColor: i === numLayers ? '#22c55e' : colors[i] }}
                                                    />
                                                    <span className="font-medium text-slate-700 dark:text-slate-200">
                                                        {i === 0 ? 'Input' : i === numLayers ? 'Output' : `After Layer ${i}`}
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="py-2 px-4 font-mono text-xs text-slate-500">
                                                {i === 0 ? 'x' : `x + ${Array.from({ length: i }).map((_, j) => `f${j + 1}`).join(' + ')}`}
                                            </td>
                                            <td className="py-2 px-4 font-mono text-slate-600 dark:text-slate-300">
                                                [{vec[0].toFixed(2)}, {vec[1].toFixed(2)}]
                                            </td>
                                            <td className="py-2 px-4">
                                                <div className="flex items-center gap-2">
                                                    <div
                                                        className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-green-400"
                                                        style={{ width: `${Math.min(100, mag * 30)}px` }}
                                                    />
                                                    <span className="font-mono text-xs text-slate-500">
                                                        {mag.toFixed(2)}
                                                    </span>
                                                </div>
                                            </td>
                                        </motion.tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Info Box */}
                <motion.div
                    className="mt-8 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-2xl p-6 text-white"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <div className="flex items-start gap-4">
                        <Info size={28} className="flex-shrink-0 mt-1" />
                        <div>
                            <h4 className="font-bold text-xl mb-2">What You're Seeing</h4>
                            <ul className="space-y-2 text-indigo-100">
                                <li className="flex items-start gap-2">
                                    <Plus size={16} className="flex-shrink-0 mt-1" />
                                    <span>
                                        Each layer adds a small vector contribution to the residual stream
                                    </span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <Layers size={16} className="flex-shrink-0 mt-1" />
                                    <span>
                                        The final output is the sum of the input plus all layer outputs: <code className="bg-white/20 px-1 rounded">x + f1 + f2 + ... + fn</code>
                                    </span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <Zap size={16} className="flex-shrink-0 mt-1" />
                                    <span>
                                        Notice how even small per-layer changes can lead to large accumulated effects.
                                        This is how depth enables complex representations!
                                    </span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
