import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Settings } from 'lucide-react';

export default function VariationsPanel() {
    const canvasRef = useRef(null);
    const [isRunning, setIsRunning] = useState(false);
    const [learningRate, setLearningRate] = useState(0.01);
    const [weightDecay, setWeightDecay] = useState(0.01);
    const [step, setStep] = useState(0);

    // Initial Positions (Same for all)
    const START_X = -2.5;
    const START_Y = 2.0;

    // Paths
    const [paths, setPaths] = useState({
        sgd: [{ x: START_X, y: START_Y }],
        rmsprop: [{ x: START_X, y: START_Y }],
        adam: [{ x: START_X, y: START_Y }],
        adamw: [{ x: START_X, y: START_Y }]
    });

    // Internal States
    const state = useRef({
        sgd: {},
        rmsprop: { v: { x: 0, y: 0 } },
        adam: { m: { x: 0, y: 0 }, v: { x: 0, y: 0 }, t: 0 },
        adamw: { m: { x: 0, y: 0 }, v: { x: 0, y: 0 }, t: 0 }
    });

    // Loss Function: Rosenbrock-ish (Banana Valley)
    // f(x,y) = (1-x)^2 + 10(y-x^2)^2
    // Gradient:
    // dx = -2(1-x) - 40x(y-x^2)
    // dy = 20(y-x^2)
    // This is very hard for SGD.
    const getGradient = (x, y) => {
        const dx = -2 * (1 - x) - 40 * x * (y - x * x);
        const dy = 20 * (y - x * x);
        return { dx, dy };
    };

    // L2 Regularization Gradient (for non-W optimizers)
    // Adds 2 * wd * w to gradient
    const getL2Grad = (w, wd) => 2 * wd * w;

    const reset = () => {
        setIsRunning(false);
        setStep(0);
        setPaths({
            sgd: [{ x: START_X, y: START_Y }],
            rmsprop: [{ x: START_X, y: START_Y }],
            adam: [{ x: START_X, y: START_Y }],
            adamw: [{ x: START_X, y: START_Y }]
        });
        state.current = {
            sgd: {},
            rmsprop: { v: { x: 0, y: 0 } },
            adam: { m: { x: 0, y: 0 }, v: { x: 0, y: 0 }, t: 0 },
            adamw: { m: { x: 0, y: 0 }, v: { x: 0, y: 0 }, t: 0 }
        };
    };

    useEffect(() => {
        if (!isRunning) return;

        const interval = setInterval(() => {
            setStep(s => s + 1);

            setPaths(prev => {
                const nextPaths = { ...prev };

                // Helper to update a specific optimizer
                const update = (name, updateFn) => {
                    const last = prev[name][prev[name].length - 1];
                    // Stop if diverged or finished
                    if (Math.abs(last.x) > 4 || Math.abs(last.y) > 4) return;

                    const grad = getGradient(last.x, last.y);
                    const next = updateFn(last, grad, state.current[name]);
                    nextPaths[name] = [...prev[name], next];
                };

                // 1. SGD
                update('sgd', (w, g) => {
                    // L2 Regularization added to gradient
                    const g_l2 = {
                        dx: g.dx + getL2Grad(w.x, weightDecay),
                        dy: g.dy + getL2Grad(w.y, weightDecay)
                    };
                    return {
                        x: w.x - learningRate * g_l2.dx,
                        y: w.y - learningRate * g_l2.dy
                    };
                });

                // 2. RMSProp
                update('rmsprop', (w, g, s) => {
                    // L2 Regularization added to gradient
                    const g_l2 = {
                        dx: g.dx + getL2Grad(w.x, weightDecay),
                        dy: g.dy + getL2Grad(w.y, weightDecay)
                    };

                    const beta = 0.9;
                    const epsilon = 1e-8;

                    s.v.x = beta * s.v.x + (1 - beta) * (g_l2.dx ** 2);
                    s.v.y = beta * s.v.y + (1 - beta) * (g_l2.dy ** 2);

                    return {
                        x: w.x - learningRate * g_l2.dx / (Math.sqrt(s.v.x) + epsilon),
                        y: w.y - learningRate * g_l2.dy / (Math.sqrt(s.v.y) + epsilon)
                    };
                });

                // 3. Adam (Standard with L2 Reg)
                update('adam', (w, g, s) => {
                    // L2 Regularization added to gradient
                    const g_l2 = {
                        dx: g.dx + getL2Grad(w.x, weightDecay),
                        dy: g.dy + getL2Grad(w.y, weightDecay)
                    };

                    const beta1 = 0.9;
                    const beta2 = 0.999;
                    const epsilon = 1e-8;
                    s.t += 1;

                    s.m.x = beta1 * s.m.x + (1 - beta1) * g_l2.dx;
                    s.m.y = beta1 * s.m.y + (1 - beta1) * g_l2.dy;

                    s.v.x = beta2 * s.v.x + (1 - beta2) * (g_l2.dx ** 2);
                    s.v.y = beta2 * s.v.y + (1 - beta2) * (g_l2.dy ** 2);

                    const mHatX = s.m.x / (1 - Math.pow(beta1, s.t));
                    const mHatY = s.m.y / (1 - Math.pow(beta1, s.t));
                    const vHatX = s.v.x / (1 - Math.pow(beta2, s.t));
                    const vHatY = s.v.y / (1 - Math.pow(beta2, s.t));

                    return {
                        x: w.x - learningRate * mHatX / (Math.sqrt(vHatX) + epsilon),
                        y: w.y - learningRate * mHatY / (Math.sqrt(vHatY) + epsilon)
                    };
                });

                // 4. AdamW (Decoupled Weight Decay)
                update('adamw', (w, g, s) => {
                    // Gradient does NOT include L2
                    const beta1 = 0.9;
                    const beta2 = 0.999;
                    const epsilon = 1e-8;
                    s.t += 1;

                    s.m.x = beta1 * s.m.x + (1 - beta1) * g.dx;
                    s.m.y = beta1 * s.m.y + (1 - beta1) * g.dy;

                    s.v.x = beta2 * s.v.x + (1 - beta2) * (g.dx ** 2);
                    s.v.y = beta2 * s.v.y + (1 - beta2) * (g.dy ** 2);

                    const mHatX = s.m.x / (1 - Math.pow(beta1, s.t));
                    const mHatY = s.m.y / (1 - Math.pow(beta1, s.t));
                    const vHatX = s.v.x / (1 - Math.pow(beta2, s.t));
                    const vHatY = s.v.y / (1 - Math.pow(beta2, s.t));

                    // Weight Decay is applied directly to weights, NOT scaled by adaptive rate
                    // w = w - lr * (update + wd * w)
                    return {
                        x: w.x - learningRate * (mHatX / (Math.sqrt(vHatX) + epsilon) + weightDecay * w.x),
                        y: w.y - learningRate * (mHatY / (Math.sqrt(vHatY) + epsilon) + weightDecay * w.y)
                    };
                });

                return nextPaths;
            });

        }, 50);

        return () => clearInterval(interval);
    }, [isRunning, learningRate, weightDecay]);

    // Drawing
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Coordinate Transform: Map [-3, 3] to canvas
        const toCanvas = (x, y) => ({
            x: (x + 3) / 6 * width,
            y: height - (y + 3) / 6 * height // Flip Y
        });

        // Clear
        ctx.fillStyle = '#0f172a'; // Slate 900
        ctx.fillRect(0, 0, width, height);

        // Draw Contours for Rosenbrock
        // f(x,y) = (1-x)^2 + 10(y-x^2)^2
        // Min is at (1,1)
        ctx.lineWidth = 1;
        for (let i = 0; i < 15; i++) {
            // Crude contour drawing by sampling grid
            // Ideally we'd use marching squares but that's complex.
            // Let's just draw points for now or simple circles? No, Rosenbrock is curved.
            // Let's just draw the target (1,1)
        }

        // Draw Target
        const target = toCanvas(1, 1);
        ctx.strokeStyle = 'white';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(target.x - 10, target.y);
        ctx.lineTo(target.x + 10, target.y);
        ctx.moveTo(target.x, target.y - 10);
        ctx.lineTo(target.x, target.y + 10);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw Paths
        const drawPath = (path, color, label) => {
            if (path.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            const start = toCanvas(path[0].x, path[0].y);
            ctx.moveTo(start.x, start.y);
            for (let i = 1; i < path.length; i++) {
                const p = toCanvas(path[i].x, path[i].y);
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();

            // Draw Head
            const last = path[path.length - 1];
            const p = toCanvas(last.x, last.y);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
            ctx.fill();
        };

        drawPath(paths.sgd, '#94a3b8', 'SGD'); // Slate 400
        drawPath(paths.rmsprop, '#facc15', 'RMSProp'); // Yellow 400
        drawPath(paths.adam, '#f472b6', 'Adam'); // Pink 400
        drawPath(paths.adamw, '#10b981', 'AdamW'); // Emerald 500

    }, [paths]);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-teal-600 dark:text-teal-400 mb-4">Adam Variations</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    Comparing <strong>SGD</strong>, <strong>RMSProp</strong>, <strong>Adam</strong>, and <strong>AdamW</strong> on the Rosenbrock function (Banana Valley).
                    <br />
                    Notice how AdamW (Green) handles weight decay differently than Adam (Pink).
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Hyperparameters</h3>

                    <div className="space-y-6 mb-8">
                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Learning Rate</label>
                                <span className="font-mono font-bold text-teal-600 dark:text-teal-400">{learningRate.toFixed(4)}</span>
                            </div>
                            <input
                                type="range" min="0.0001" max="0.02" step="0.0001"
                                value={learningRate}
                                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                className="w-full accent-teal-400"
                            />
                        </div>

                        <div>
                            <div className="flex justify-between items-end mb-2">
                                <label className="text-sm text-slate-800 dark:text-slate-400">Weight Decay (L2)</label>
                                <span className="font-mono font-bold text-teal-600 dark:text-teal-400">{weightDecay.toFixed(4)}</span>
                            </div>
                            <input
                                type="range" min="0" max="0.1" step="0.001"
                                value={weightDecay}
                                onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                                className="w-full accent-teal-400"
                            />
                        </div>
                    </div>

                    <div className="flex gap-4 mb-8">
                        <button
                            onClick={() => setIsRunning(!isRunning)}
                            className={`flex-1 py-3 rounded-xl font-bold shadow-lg transition-all flex items-center justify-center gap-2 ${isRunning
                                    ? 'bg-slate-700 text-slate-700 dark:text-slate-300'
                                    : 'bg-teal-600 hover:bg-teal-500 text-white'
                                }`}
                        >
                            <Play size={20} />
                            {isRunning ? 'Pause' : 'Start Race'}
                        </button>
                        <button
                            onClick={reset}
                            className="px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-xl font-bold shadow-lg transition-all"
                        >
                            <RotateCcw size={20} />
                        </button>
                    </div>

                    <div className="bg-slate-900 p-4 rounded-lg text-sm text-slate-700 dark:text-slate-300 space-y-3">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-slate-400"></div>
                            <span><strong>SGD:</strong> Often gets stuck in the curve.</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                            <span><strong>RMSProp:</strong> Adapts to curvature, faster than SGD.</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-pink-400"></div>
                            <span><strong>Adam:</strong> Momentum + RMSProp. Fast convergence.</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                            <span><strong>AdamW:</strong> Decoupled Weight Decay. The LLM standard.</span>
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 flex flex-col items-center">
                    <canvas
                        ref={canvasRef}
                        width={400}
                        height={400}
                        className="bg-slate-900 rounded-lg border border-slate-600 w-full max-w-[400px]"
                    />
                    <div className="mt-4 text-center text-slate-800 dark:text-sm">
                        Target is at (1, 1). Start is at (-2.5, 2.0).
                        <br />
                        The "Banana Valley" is curved and narrow.
                    </div>
                </div>
            </div>
        </div>
    );
}
