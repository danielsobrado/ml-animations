import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw } from 'lucide-react';

export default function DescentPanel() {
    const canvasRef = useRef(null);
    const [isRunning, setIsRunning] = useState(false);
    const [learningRate, setLearningRate] = useState(0.02);
    const [step, setStep] = useState(0);

    // Initial Positions
    const START_X = -2.0;
    const START_Y = 1.5;

    // State for optimizers
    const [sgdPath, setSgdPath] = useState([{ x: START_X, y: START_Y }]);
    const [adamPath, setAdamPath] = useState([{ x: START_X, y: START_Y }]);

    // Adam Internal State
    const adamState = useRef({
        m: { x: 0, y: 0 },
        v: { x: 0, y: 0 },
        t: 0
    });

    // Loss Function: f(x,y) = 0.1x^2 + 2y^2 (Elongated Valley)
    // Gradient: dx = 0.2x, dy = 4y
    const getGradient = (x, y) => ({ dx: 0.2 * x, dy: 4 * y });

    const reset = () => {
        setIsRunning(false);
        setStep(0);
        setSgdPath([{ x: START_X, y: START_Y }]);
        setAdamPath([{ x: START_X, y: START_Y }]);
        adamState.current = { m: { x: 0, y: 0 }, v: { x: 0, y: 0 }, t: 0 };
    };

    useEffect(() => {
        if (!isRunning) return;

        const interval = setInterval(() => {
            setStep(s => s + 1);

            // SGD Update
            setSgdPath(prev => {
                const last = prev[prev.length - 1];
                const grad = getGradient(last.x, last.y);
                // Standard SGD: w = w - lr * grad
                return [...prev, {
                    x: last.x - learningRate * grad.dx,
                    y: last.y - learningRate * grad.dy
                }];
            });

            // Adam Update
            setAdamPath(prev => {
                const last = prev[prev.length - 1];
                const grad = getGradient(last.x, last.y);
                const { m, v, t } = adamState.current;

                const beta1 = 0.9;
                const beta2 = 0.999;
                const epsilon = 1e-8;
                const newT = t + 1;

                // Update moments
                const newM = {
                    x: beta1 * m.x + (1 - beta1) * grad.dx,
                    y: beta1 * m.y + (1 - beta1) * grad.dy
                };
                const newV = {
                    x: beta2 * v.x + (1 - beta2) * (grad.dx ** 2),
                    y: beta2 * v.y + (1 - beta2) * (grad.dy ** 2)
                };

                // Bias correction
                const mHat = {
                    x: newM.x / (1 - Math.pow(beta1, newT)),
                    y: newM.y / (1 - Math.pow(beta1, newT))
                };
                const vHat = {
                    x: newV.x / (1 - Math.pow(beta2, newT)),
                    y: newV.y / (1 - Math.pow(beta2, newT))
                };

                // Update state
                adamState.current = { m: newM, v: newV, t: newT };

                // Adam Step
                return [...prev, {
                    x: last.x - learningRate * mHat.x / (Math.sqrt(vHat.x) + epsilon),
                    y: last.y - learningRate * mHat.y / (Math.sqrt(vHat.y) + epsilon)
                }];
            });

        }, 50);

        return () => clearInterval(interval);
    }, [isRunning, learningRate]);

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

        // Draw Contours
        ctx.lineWidth = 1;
        for (let i = 0; i < 20; i++) {
            const val = i * i * 0.5; // Contour levels
            ctx.strokeStyle = `rgba(255, 255, 255, 0.1)`;
            ctx.beginPath();
            // Ellipse approximation for 0.1x^2 + 2y^2 = val
            // x^2 / (val/0.1) + y^2 / (val/2) = 1
            // a = sqrt(10*val), b = sqrt(0.5*val)
            if (val > 0) {
                ctx.ellipse(
                    width / 2, height / 2,
                    Math.sqrt(10 * val) * (width / 6),
                    Math.sqrt(0.5 * val) * (height / 6),
                    0, 0, 2 * Math.PI
                );
            }
            ctx.stroke();
        }

        // Draw Paths
        const drawPath = (path, color) => {
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

        drawPath(sgdPath, '#ef4444'); // Red for SGD
        drawPath(adamPath, '#10b981'); // Emerald for Adam

    }, [sgdPath, adamPath]);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400 mb-4">SGD vs Adam</h2>
                <p className="text-lg text-slate-700 dark:text-slate-300 leading-relaxed">
                    <strong>SGD (Red)</strong>: Gets stuck bouncing back and forth in steep valleys (jittery).
                    <br />
                    <strong>Adam (Green)</strong>: Uses <em>Momentum</em> and <em>Adaptive Rates</em> to smooth out the path and accelerate.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Simulation Controls</h3>

                    <div className="mb-6">
                        <div className="flex justify-between items-end mb-2">
                            <label className="text-sm text-slate-800 dark:text-slate-400">Learning Rate</label>
                            <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">{learningRate.toFixed(3)}</span>
                        </div>
                        <input
                            type="range" min="0.001" max="0.1" step="0.001"
                            value={learningRate}
                            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                            className="w-full accent-emerald-400"
                        />
                    </div>

                    <div className="flex gap-4 mb-8">
                        <button
                            onClick={() => setIsRunning(!isRunning)}
                            className={`flex-1 py-3 rounded-xl font-bold shadow-lg transition-all flex items-center justify-center gap-2 ${isRunning
                                    ? 'bg-slate-700 text-slate-700 dark:text-slate-300'
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
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

                    <div className="bg-slate-900 p-4 rounded-lg text-sm text-slate-700 dark:text-slate-300 space-y-2">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <span><strong>SGD:</strong> Simple gradient descent. Struggles with different scales (steep Y, shallow X).</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                            <span><strong>Adam:</strong> Adaptive Moment Estimation. Scales updates individually for X and Y.</span>
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
                        Top-down view of the Loss Landscape (Contour Plot)
                    </div>
                </div>
            </div>
        </div>
    );
}
