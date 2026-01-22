import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Info, ChevronRight } from 'lucide-react';

export default function EulerSchedulerPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [numSteps, setNumSteps] = useState(10);
  const [solver, setSolver] = useState('euler');
  const canvasRef = useRef(null);

  const solvers = {
    euler: {
      name: 'Euler (1st Order)',
      formula: 'x_{t+Δt} = x_t + Δt · v_θ(x_t, t)',
      description: 'Simplest ODE solver. Takes one velocity evaluation per step.',
      color: '#ec4899',
      order: 1
    },
    heun: {
      name: 'Heun (2nd Order)',
      formula: 'x̃ = x_t + Δt·v(x_t, t); x_{t+Δt} = x_t + Δt/2·(v(x_t, t) + v(x̃, t+Δt))',
      description: 'Predictor-corrector method. More accurate but needs 2 evaluations.',
      color: '#8b5cf6',
      order: 2
    }
  };

  // Animation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setStep(prev => {
        if (prev >= numSteps) {
          setIsPlaying(false);
          return numSteps;
        }
        return prev + 1;
      });
    }, 800);
    return () => clearInterval(interval);
  }, [isPlaying, numSteps]);

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, width, height);

    // Define the "true" path (curved for demonstration)
    const truePath = (t) => {
      const x = 60 + t * (width - 120);
      // Add a curve to show solver accuracy difference
      const y = height / 2 + Math.sin(t * Math.PI * 2) * 60 - t * 40;
      return { x, y };
    };

    // Draw true path
    ctx.beginPath();
    ctx.moveTo(truePath(0).x, truePath(0).y);
    for (let t = 0; t <= 1; t += 0.02) {
      const p = truePath(t);
      ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Simulate solver paths
    const dt = 1 / numSteps;
    const currentSteps = Math.min(step, numSteps);

    // Euler path simulation (with accumulated error)
    let eulerX = truePath(0).x;
    let eulerY = truePath(0).y;
    const eulerPath = [{ x: eulerX, y: eulerY, t: 0 }];

    for (let i = 0; i < currentSteps; i++) {
      const t = i / numSteps;
      const nextT = (i + 1) / numSteps;
      
      // Get velocity at current point (derivative of true path)
      const curr = truePath(t);
      const next = truePath(t + 0.01);
      const vx = (next.x - curr.x) / 0.01;
      const vy = (next.y - curr.y) / 0.01;

      // Euler step (from current solver position, not true position)
      const scale = dt / 0.01;
      eulerX += vx * scale * 0.8; // Add some error for demonstration
      eulerY += vy * scale * 0.8;

      eulerPath.push({ x: eulerX, y: eulerY, t: nextT });
    }

    // Heun path simulation (more accurate)
    let heunX = truePath(0).x;
    let heunY = truePath(0).y;
    const heunPath = [{ x: heunX, y: heunY, t: 0 }];

    for (let i = 0; i < currentSteps; i++) {
      const t = i / numSteps;
      
      // Velocity at current position
      const curr = truePath(t);
      const next = truePath(t + 0.01);
      const vx1 = (next.x - curr.x) / 0.01;
      const vy1 = (next.y - curr.y) / 0.01;

      // Predictor step
      const predictX = heunX + vx1 * dt / 0.01;
      const predictY = heunY + vy1 * dt / 0.01;

      // Velocity at predicted position
      const predT = (i + 1) / numSteps;
      const currPred = truePath(predT);
      const nextPred = truePath(predT + 0.01);
      const vx2 = (nextPred.x - currPred.x) / 0.01;
      const vy2 = (nextPred.y - currPred.y) / 0.01;

      // Corrector step (average velocities)
      heunX += ((vx1 + vx2) / 2) * dt / 0.01 * 0.95;
      heunY += ((vy1 + vy2) / 2) * dt / 0.01 * 0.95;

      heunPath.push({ x: heunX, y: heunY, t: predT });
    }

    // Draw solver paths
    const drawPath = (path, color, isSelected) => {
      if (path.length < 2) return;

      // Draw path segments
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x, path[i].y);
      }
      ctx.strokeStyle = isSelected ? color : `${color}60`;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.stroke();

      // Draw step points
      path.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, isSelected ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? color : `${color}80`;
        ctx.fill();

        // Draw step number
        if (isSelected && i > 0) {
          ctx.fillStyle = 'white';
          ctx.font = '10px sans-serif';
          ctx.fillText(i.toString(), p.x + 8, p.y - 8);
        }
      });
    };

    if (solver === 'euler') {
      drawPath(heunPath, solvers.heun.color, false);
      drawPath(eulerPath, solvers.euler.color, true);
    } else {
      drawPath(eulerPath, solvers.euler.color, false);
      drawPath(heunPath, solvers.heun.color, true);
    }

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '12px sans-serif';
    ctx.fillText('Start (t=0)', 40, height - 20);
    ctx.fillText('End (t=1)', width - 80, height - 20);
    ctx.fillText('True path', 20, 30);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.fillText('(dashed)', 80, 30);

    // Legend
    ctx.fillStyle = solvers.euler.color;
    ctx.fillRect(width - 150, 20, 12, 12);
    ctx.fillStyle = 'white';
    ctx.fillText('Euler', width - 130, 30);

    ctx.fillStyle = solvers.heun.color;
    ctx.fillRect(width - 150, 40, 12, 12);
    ctx.fillStyle = 'white';
    ctx.fillText('Heun', width - 130, 50);

  }, [step, numSteps, solver]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Euler Scheduler: <span className="text-fuchsia-600 dark:text-fuchsia-400">ODE Solvers</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Numerical methods to follow the velocity field from noise to data
        </p>
      </div>

      {/* Solver Selection */}
      <div className="grid md:grid-cols-2 gap-4">
        {Object.entries(solvers).map(([key, s]) => (
          <button
            key={key}
            onClick={() => setSolver(key)}
            className={`p-4 rounded-xl border transition-all text-left ${
              solver === key
                ? 'border-fuchsia-500 bg-fuchsia-500/20'
                : 'border-white/10 bg-black/30 hover:bg-white/5'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: s.color }} />
              <span className="font-bold">{s.name}</span>
              <span className="text-xs bg-white/10 px-2 py-0.5 rounded">Order {s.order}</span>
            </div>
            <p className="text-xs text-gray-800 dark:text-gray-400 mb-2">{s.description}</p>
            <code className="text-xs text-gray-700 dark:text-gray-500 block overflow-x-auto">{s.formula}</code>
          </button>
        ))}
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <canvas
          ref={canvasRef}
          width={700}
          height={300}
          className="w-full rounded-xl mb-4"
        />

        {/* Controls */}
        <div className="flex flex-wrap justify-center gap-4 mb-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-2 px-4 py-2 bg-fuchsia-600 hover:bg-fuchsia-700 rounded-lg transition-colors"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Run Steps'}
          </button>
          <button
            onClick={() => { setStep(0); setIsPlaying(false); }}
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
          <button
            onClick={() => setStep(prev => Math.min(prev + 1, numSteps))}
            disabled={step >= numSteps}
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors disabled:opacity-50"
          >
            <ChevronRight size={18} />
            Single Step
          </button>
        </div>

        {/* Step Count Slider */}
        <div className="flex items-center gap-4 px-4">
          <span className="text-sm text-gray-800 dark:text-gray-400 w-20">Steps: {numSteps}</span>
          <input
            type="range"
            min="4"
            max="30"
            step="1"
            value={numSteps}
            onChange={(e) => {
              setNumSteps(parseInt(e.target.value));
              setStep(0);
              setIsPlaying(false);
            }}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Step Counter */}
        <div className="text-center mt-4 text-lg">
          Step <span className="text-fuchsia-600 dark:text-fuchsia-400 font-bold">{step}</span> / {numSteps}
        </div>
      </div>

      {/* Euler Method Explained */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Info size={20} className="text-fuchsia-600 dark:text-fuchsia-400" />
          Euler Method Deep Dive
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-bold text-fuchsia-300 mb-3">The Algorithm</h4>
            <div className="bg-black/30 rounded-lg p-4 font-mono text-sm space-y-2">
              <p className="text-gray-800 dark:text-gray-400"># Initialize</p>
              <p><span className="text-blue-600 dark:text-blue-400">x</span> = sample_noise()</p>
              <p><span className="text-blue-600 dark:text-blue-400">dt</span> = 1.0 / num_steps</p>
              <p className="text-gray-800 dark:text-gray-400 mt-2"># Iterate</p>
              <p><span className="text-purple-600 dark:text-purple-400">for</span> t <span className="text-purple-600 dark:text-purple-400">in</span> linspace(0, 1, num_steps):</p>
              <p className="pl-4"><span className="text-blue-600 dark:text-blue-400">v</span> = model(x, t)</p>
              <p className="pl-4"><span className="text-blue-600 dark:text-blue-400">x</span> = x + v * dt</p>
              <p className="text-gray-800 dark:text-gray-400 mt-2"># x is now the generated sample</p>
            </div>
          </div>
          <div>
            <h4 className="font-bold text-purple-300 mb-3">Key Insights</h4>
            <ul className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-fuchsia-600 dark:text-fuchsia-400">•</span>
                <span><strong>Discretization:</strong> We approximate the continuous ODE with discrete steps</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-fuchsia-600 dark:text-fuchsia-400">•</span>
                <span><strong>Error:</strong> More steps = less error, but more compute</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-fuchsia-600 dark:text-fuchsia-400">•</span>
                <span><strong>Trade-off:</strong> SD3 typically uses 20-50 steps for good quality</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-fuchsia-600 dark:text-fuchsia-400">•</span>
                <span><strong>Heun advantage:</strong> 2nd order means same accuracy with fewer steps</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Error Analysis */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Solver Accuracy Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Solver</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Order</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">NFE/step</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Error (O)</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Best For</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-fuchsia-600 dark:text-fuchsia-400">Euler</td>
                <td className="py-3 px-4">1st</td>
                <td className="py-3 px-4">1</td>
                <td className="py-3 px-4">O(Δt)</td>
                <td className="py-3 px-4">Fast prototyping, many steps</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-purple-600 dark:text-purple-400">Heun</td>
                <td className="py-3 px-4">2nd</td>
                <td className="py-3 px-4">2</td>
                <td className="py-3 px-4">O(Δt²)</td>
                <td className="py-3 px-4">Balanced quality/speed</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">DPM++</td>
                <td className="py-3 px-4">2nd+</td>
                <td className="py-3 px-4">1-2</td>
                <td className="py-3 px-4">O(Δt²)</td>
                <td className="py-3 px-4">Production, few steps</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-700 dark:text-gray-500 mt-4">
          NFE = Number of Function Evaluations (model calls). Higher order solvers need more evaluations but achieve better accuracy per step.
        </p>
      </div>
    </div>
  );
}
