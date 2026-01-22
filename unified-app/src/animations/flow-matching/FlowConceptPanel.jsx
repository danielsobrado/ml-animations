import React, { useState, useEffect, useRef } from 'react';
import { ArrowRight, Play, Pause, RotateCcw, Info } from 'lucide-react';

export default function FlowConceptPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [selectedPath, setSelectedPath] = useState('linear');
  const canvasRef = useRef(null);

  const paths = {
    linear: {
      name: 'Linear (Optimal Transport)',
      formula: 'x_t = (1-t)·x₀ + t·x₁',
      description: 'Straight-line interpolation from noise to data. Simple and efficient.',
      color: '#ec4899'
    },
    cosine: {
      name: 'Cosine Schedule',
      formula: 'x_t = cos²(πt/2)·x₀ + sin²(πt/2)·x₁',
      description: 'Smoother transition, spending more time near data distribution.',
      color: '#8b5cf6'
    },
    variance: {
      name: 'Variance Preserving',
      formula: 'x_t = √(1-σ²_t)·x₁ + σ_t·x₀',
      description: 'Maintains unit variance throughout, similar to DDPM.',
      color: '#3b82f6'
    }
  };

  // Animation timer
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setTime(prev => {
        if (prev >= 1) {
          setIsPlaying(false);
          return 1;
        }
        return prev + 0.02;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Define noise samples (left) and data samples (right)
    const noisePoints = [
      { x: 80, y: 80 },
      { x: 60, y: 200 },
      { x: 100, y: 280 },
      { x: 40, y: 160 },
    ];

    const dataPoints = [
      { x: 520, y: 120 },
      { x: 500, y: 200 },
      { x: 540, y: 260 },
      { x: 480, y: 180 },
    ];

    // Interpolation function based on selected path
    const interpolate = (x0, x1, t) => {
      if (selectedPath === 'linear') {
        return (1 - t) * x0 + t * x1;
      } else if (selectedPath === 'cosine') {
        const w0 = Math.cos(Math.PI * t / 2) ** 2;
        const w1 = Math.sin(Math.PI * t / 2) ** 2;
        return w0 * x0 + w1 * x1;
      } else {
        // Variance preserving (simplified)
        const sigma = 1 - t;
        return Math.sqrt(1 - sigma * sigma) * x1 + sigma * x0;
      }
    };

    // Draw paths and current positions
    noisePoints.forEach((noise, i) => {
      const data = dataPoints[i];
      const color = paths[selectedPath].color;

      // Draw full path (faded)
      ctx.beginPath();
      ctx.moveTo(noise.x, noise.y);
      for (let t = 0; t <= 1; t += 0.05) {
        const x = interpolate(noise.x, data.x, t);
        const y = interpolate(noise.y, data.y, t);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `${color}40`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw completed path (solid)
      if (time > 0) {
        ctx.beginPath();
        ctx.moveTo(noise.x, noise.y);
        for (let t = 0; t <= time; t += 0.05) {
          const x = interpolate(noise.x, data.x, t);
          const y = interpolate(noise.y, data.y, t);
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Draw starting point (noise)
      ctx.beginPath();
      ctx.arc(noise.x, noise.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(100, 100, 100, 0.8)';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw ending point (data)
      ctx.beginPath();
      ctx.arc(data.x, data.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw current position
      const currentX = interpolate(noise.x, data.x, time);
      const currentY = interpolate(noise.y, data.y, time);
      ctx.beginPath();
      ctx.arc(currentX, currentY, 10, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Draw velocity arrow at current position
      const dt = 0.01;
      const nextX = interpolate(noise.x, data.x, Math.min(time + dt, 1));
      const nextY = interpolate(noise.y, data.y, Math.min(time + dt, 1));
      const vx = (nextX - currentX) / dt * 0.3;
      const vy = (nextY - currentY) / dt * 0.3;

      if (time < 0.99) {
        ctx.beginPath();
        ctx.moveTo(currentX, currentY);
        ctx.lineTo(currentX + vx, currentY + vy);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(vy, vx);
        ctx.beginPath();
        ctx.moveTo(currentX + vx, currentY + vy);
        ctx.lineTo(currentX + vx - 8 * Math.cos(angle - 0.4), currentY + vy - 8 * Math.sin(angle - 0.4));
        ctx.lineTo(currentX + vx - 8 * Math.cos(angle + 0.4), currentY + vy - 8 * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
      }
    });

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '14px sans-serif';
    ctx.fillText('Noise (t=0)', 20, 40);
    ctx.fillText('Data (t=1)', 480, 40);
    ctx.fillText(`t = ${time.toFixed(2)}`, width / 2 - 30, 40);

  }, [time, selectedPath]);

  const resetAnimation = () => {
    setTime(0);
    setIsPlaying(false);
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          The Flow: <span className="text-fuchsia-600 dark:text-fuchsia-400">Probability Transport</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Watch samples flow from noise to data along learned paths
        </p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <canvas
          ref={canvasRef}
          width={600}
          height={340}
          className="w-full rounded-xl bg-black/30 mb-4"
        />

        {/* Controls */}
        <div className="flex justify-center gap-4 mb-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-2 px-4 py-2 bg-fuchsia-600 hover:bg-fuchsia-700 rounded-lg transition-colors"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={resetAnimation}
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {/* Time Slider */}
        <div className="flex items-center gap-4 px-4">
          <span className="text-sm text-gray-800 dark:text-gray-400 w-12">t = 0</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={time}
            onChange={(e) => {
              setTime(parseFloat(e.target.value));
              setIsPlaying(false);
            }}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-sm text-gray-800 dark:text-gray-400 w-12">t = 1</span>
        </div>
      </div>

      {/* Path Selector */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Choose Interpolation Path</h3>
        <div className="grid md:grid-cols-3 gap-4">
          {Object.entries(paths).map(([key, path]) => (
            <button
              key={key}
              onClick={() => setSelectedPath(key)}
              className={`p-4 rounded-xl border transition-all text-left ${
                selectedPath === key
                  ? 'border-fuchsia-500 bg-fuchsia-500/20'
                  : 'border-white/10 bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: path.color }}
                />
                <span className="font-bold">{path.name}</span>
              </div>
              <code className="text-sm text-gray-800 dark:text-gray-400 block mb-2">{path.formula}</code>
              <p className="text-xs text-gray-700 dark:text-gray-500">{path.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Conditional Flow Matching */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Info size={20} className="text-fuchsia-600 dark:text-fuchsia-400" />
          Conditional Flow Matching (CFM)
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-bold text-fuchsia-300 mb-2">The Training Trick</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              We can't compute the true marginal velocity field v(x_t, t) directly. Instead, 
              we condition on individual pairs (x₀, x₁) and learn the conditional velocity:
            </p>
            <div className="bg-black/30 rounded-lg p-3 font-mono text-center text-fuchsia-300">
              v_θ(x_t, t) ≈ (x₁ - x₀)
            </div>
          </div>
          <div>
            <h4 className="font-bold text-purple-300 mb-2">Training Algorithm</h4>
            <ol className="text-sm text-gray-700 dark:text-gray-300 space-y-2 list-decimal list-inside">
              <li>Sample x₀ ~ N(0, I) (noise)</li>
              <li>Sample x₁ ~ p_data (real data)</li>
              <li>Sample t ~ U(0, 1)</li>
              <li>Compute x_t = (1-t)x₀ + tx₁</li>
              <li>Target velocity: v* = x₁ - x₀</li>
              <li>Loss: ||v_θ(x_t, t) - v*||²</li>
            </ol>
          </div>
        </div>
      </div>

      {/* Why This Works */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Why This Works: Intuition</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-fuchsia-600 rounded-full flex items-center justify-center text-sm font-bold">1</div>
              <div>
                <h4 className="font-bold">Straight Paths Are Simple</h4>
                <p className="text-sm text-gray-800 dark:text-gray-400">
                  Linear interpolation x_t = (1-t)x₀ + tx₁ creates the simplest possible path.
                  The velocity is constant: v = x₁ - x₀.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-sm font-bold">2</div>
              <div>
                <h4 className="font-bold">Network Learns Average</h4>
                <p className="text-sm text-gray-800 dark:text-gray-400">
                  At each (x_t, t), multiple flows pass through. The network learns the 
                  average velocity direction, which transports probability mass correctly.
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold">3</div>
              <div>
                <h4 className="font-bold">Consistent Velocity Field</h4>
                <p className="text-sm text-gray-800 dark:text-gray-400">
                  Even though we train on random pairs, the learned velocity field becomes 
                  consistent - it always points "toward" the data distribution.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-cyan-600 rounded-full flex items-center justify-center text-sm font-bold">4</div>
              <div>
                <h4 className="font-bold">ODE Solver Follows</h4>
                <p className="text-sm text-gray-800 dark:text-gray-400">
                  At generation time, we start from noise and use an ODE solver (Euler, Heun) 
                  to follow the learned velocity field until we reach the data.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
