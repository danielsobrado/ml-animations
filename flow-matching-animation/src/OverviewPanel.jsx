import React, { useState, useEffect, useRef } from 'react';
import { ArrowRight, Info, Zap, Shuffle, Target, Play, Pause, RotateCcw } from 'lucide-react';

export default function OverviewPanel() {
  const [hoveredComponent, setHoveredComponent] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [flowProgress, setFlowProgress] = useState(0);
  const canvasRef = useRef(null);

  // Animate particles flowing from noise to data
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setFlowProgress(prev => (prev + 1) % 100);
    }, 50);
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Draw flow field visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw flow field arrows
    const gridSize = 30;
    for (let x = gridSize; x < width; x += gridSize) {
      for (let y = gridSize; y < height; y += gridSize) {
        // Flow direction: from edges (noise) toward center (data)
        const centerX = width / 2;
        const centerY = height / 2;
        const dx = (centerX - x) / width;
        const dy = (centerY - y) / height;
        const magnitude = Math.sqrt(dx * dx + dy * dy);

        // Normalize and scale
        const arrowLength = 15;
        const nx = (dx / magnitude) * arrowLength;
        const ny = (dy / magnitude) * arrowLength;

        // Draw arrow
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + nx, y + ny);
        ctx.strokeStyle = `rgba(236, 72, 153, ${0.3 + magnitude * 0.5})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Arrow head
        const headSize = 4;
        const angle = Math.atan2(ny, nx);
        ctx.beginPath();
        ctx.moveTo(x + nx, y + ny);
        ctx.lineTo(
          x + nx - headSize * Math.cos(angle - Math.PI / 6),
          y + ny - headSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          x + nx - headSize * Math.cos(angle + Math.PI / 6),
          y + ny - headSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = `rgba(236, 72, 153, ${0.3 + magnitude * 0.5})`;
        ctx.fill();
      }
    }

    // Draw animated particles following the flow
    const particleCount = 20;
    for (let i = 0; i < particleCount; i++) {
      const phase = (flowProgress + i * (100 / particleCount)) % 100;
      const t = phase / 100;

      // Start from random edge positions, flow toward center
      const angle = (i / particleCount) * Math.PI * 2;
      const startX = width / 2 + Math.cos(angle) * (width / 2 - 20);
      const startY = height / 2 + Math.sin(angle) * (height / 2 - 20);
      const endX = width / 2;
      const endY = height / 2;

      const x = startX + (endX - startX) * t;
      const y = startY + (endY - startY) * t;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(139, 92, 246, ${1 - t * 0.5})`;
      ctx.fill();
    }
  }, [flowProgress]);

  const components = {
    noise: {
      title: 'Noise Distribution p₀',
      description: 'Starting point: pure Gaussian noise. Data has no structure, just random samples from N(0, I).',
      color: 'from-gray-500 to-slate-500',
      details: ['z₀ ~ N(0, I)', 'No data structure', 'High entropy']
    },
    flow: {
      title: 'Velocity Field v_θ',
      description: 'Neural network predicts the velocity at each point to transport noise toward data.',
      color: 'from-fuchsia-500 to-purple-500',
      details: ['Learned by network', 'Time-dependent', 'Defines ODE flow']
    },
    data: {
      title: 'Data Distribution p₁',
      description: 'Target: the distribution of real data (images). We want to learn to generate samples from this.',
      color: 'from-blue-500 to-cyan-500',
      details: ['Real images', 'Complex structure', 'Low entropy']
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Flow Matching: <span className="text-fuchsia-400">The Big Picture</span>
        </h2>
        <p className="text-gray-400">
          Learning continuous probability flows from noise to data
        </p>
      </div>

      {/* Key Insight Box */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <div className="flex items-start gap-4">
          <Zap className="text-fuchsia-400 mt-1" size={24} />
          <div>
            <h3 className="font-bold text-lg text-fuchsia-300 mb-2">The Core Idea</h3>
            <p className="text-gray-300">
              Instead of learning to <strong>denoise</strong> step by step (like DDPM), flow matching learns a 
              <strong className="text-fuchsia-400"> continuous velocity field</strong> that transports samples 
              from noise to data. Think of it as learning the "current" that carries particles through probability space.
            </p>
          </div>
        </div>
      </div>

      {/* Main Flow Diagram */}
      <div className="bg-black/30 rounded-2xl p-8 border border-white/10">
        <div className="flex items-center justify-between gap-4 mb-6">
          {/* Noise */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('noise')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-28 h-28 rounded-xl bg-gradient-to-br ${components.noise.color} flex items-center justify-center relative overflow-hidden`}>
              <div className="absolute inset-0 flex flex-wrap gap-1 p-2 opacity-60">
                {Array(16).fill(0).map((_, i) => (
                  <div key={i} className="w-4 h-4 bg-white/40 rounded-sm" style={{
                    opacity: Math.random() * 0.8 + 0.2
                  }} />
                ))}
              </div>
              <div className="relative z-10 text-center">
                <p className="text-lg font-bold">p₀</p>
                <p className="text-xs opacity-80">Noise</p>
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">t = 0</p>
          </div>

          {/* Flow Field */}
          <div className="flex-1 flex items-center justify-center">
            <div
              className="relative cursor-pointer"
              onMouseEnter={() => setHoveredComponent('flow')}
              onMouseLeave={() => setHoveredComponent(null)}
            >
              <canvas 
                ref={canvasRef} 
                width={300} 
                height={150}
                className="rounded-xl bg-black/20"
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-black/50 px-3 py-1 rounded-lg">
                  <p className="text-sm font-mono text-fuchsia-300">v_θ(x_t, t)</p>
                </div>
              </div>
            </div>
          </div>

          {/* Data */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('data')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-28 h-28 rounded-xl bg-gradient-to-br ${components.data.color} flex items-center justify-center relative overflow-hidden`}>
              <div className="absolute inset-0 flex items-center justify-center opacity-40">
                <div className="w-16 h-16 rounded-lg bg-white/50" />
              </div>
              <div className="relative z-10 text-center">
                <p className="text-lg font-bold">p₁</p>
                <p className="text-xs opacity-80">Data</p>
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">t = 1</p>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex justify-center gap-4 mb-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-2 px-4 py-2 bg-fuchsia-600 hover:bg-fuchsia-700 rounded-lg transition-colors"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => setFlowProgress(0)}
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {/* Hover Info Panel */}
        {hoveredComponent && (
          <div className="mt-4 p-4 bg-white/5 rounded-xl border border-white/10">
            <div className="flex items-start gap-3">
              <Info className="text-fuchsia-400 mt-1" size={20} />
              <div>
                <h4 className="font-bold text-lg">{components[hoveredComponent].title}</h4>
                <p className="text-gray-300 mt-1">{components[hoveredComponent].description}</p>
                <div className="flex gap-2 mt-3 flex-wrap">
                  {components[hoveredComponent].details.map((detail, i) => (
                    <span key={i} className="bg-white/10 px-2 py-1 rounded text-sm">{detail}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* The ODE Equation */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 text-center">The Flow ODE</h3>
        <div className="text-center text-2xl font-mono text-fuchsia-300 mb-4">
          dx/dt = v_θ(x_t, t)
        </div>
        <p className="text-gray-400 text-center max-w-2xl mx-auto">
          Starting from x₀ ~ N(0, I), we integrate this ODE from t=0 to t=1 to get x₁ ~ p_data. 
          The network v_θ learns the velocity field that makes this transformation happen.
        </p>
      </div>

      {/* Key Concepts Grid */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-fuchsia-900/30 to-fuchsia-800/20 rounded-xl p-5 border border-fuchsia-500/30">
          <h3 className="font-bold text-fuchsia-300 mb-2 flex items-center gap-2">
            <Zap size={18} /> Why "Flow"?
          </h3>
          <p className="text-sm text-gray-300">
            Inspired by physics: particles flow through a velocity field. Here, probability mass 
            "flows" from the noise distribution to the data distribution along learned paths.
          </p>
        </div>

        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-2 flex items-center gap-2">
            <Shuffle size={18} /> vs DDPM
          </h3>
          <p className="text-sm text-gray-300">
            DDPM predicts noise ε to be removed. Flow matching predicts velocity v directly.
            Both achieve similar results, but flow matching has simpler training objectives.
          </p>
        </div>

        <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-xl p-5 border border-blue-500/30">
          <h3 className="font-bold text-blue-300 mb-2 flex items-center gap-2">
            <Target size={18} /> SD3's Choice
          </h3>
          <p className="text-sm text-gray-300">
            Stable Diffusion 3 uses flow matching because it enables more flexible noise schedules
            and often converges faster with fewer sampling steps.
          </p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 text-center">Flow Matching vs Score Matching (DDPM)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-400">Aspect</th>
                <th className="py-3 px-4 text-left text-fuchsia-400">Flow Matching</th>
                <th className="py-3 px-4 text-left text-blue-400">Score Matching (DDPM)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-gray-300">Network predicts</td>
                <td className="py-3 px-4">Velocity v_θ(x_t, t)</td>
                <td className="py-3 px-4">Noise ε_θ(x_t, t)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-300">Training objective</td>
                <td className="py-3 px-4">MSE on velocity</td>
                <td className="py-3 px-4">MSE on noise</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-300">Interpolation path</td>
                <td className="py-3 px-4">Linear: x_t = (1-t)x₀ + tx₁</td>
                <td className="py-3 px-4">Gaussian: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-300">Sampling</td>
                <td className="py-3 px-4">ODE solver (Euler, Heun)</td>
                <td className="py-3 px-4">SDE/ODE (DDPM, DDIM)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
