import React, { useState, useEffect, useRef } from 'react';
import { Info } from 'lucide-react';

export default function SigmaSchedulePanel() {
  const [selectedSchedule, setSelectedSchedule] = useState('linear');
  const [numSteps, setNumSteps] = useState(20);
  const canvasRef = useRef(null);

  const schedules = {
    linear: {
      name: 'Linear',
      formula: 'σ(t) = 1 - t',
      description: 'Constant denoising rate. Simple and effective baseline.',
      color: '#ec4899',
      fn: (t) => 1 - t
    },
    cosine: {
      name: 'Cosine',
      formula: 'σ(t) = cos(πt/2)',
      description: 'Slower start and end, faster in the middle. Better for high-res images.',
      color: '#8b5cf6',
      fn: (t) => Math.cos(Math.PI * t / 2)
    },
    karras: {
      name: 'Karras',
      formula: 'σ(t) = (σ_max^(1/ρ) + t(σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ',
      description: 'Optimized schedule from Karras et al. Popular in practice (ρ=7).',
      color: '#3b82f6',
      fn: (t) => {
        const sigmaMax = 1;
        const sigmaMin = 0.002;
        const rho = 7;
        const invRho = 1 / rho;
        return Math.pow(
          Math.pow(sigmaMax, invRho) + t * (Math.pow(sigmaMin, invRho) - Math.pow(sigmaMax, invRho)),
          rho
        );
      }
    },
    exponential: {
      name: 'Exponential',
      formula: 'σ(t) = σ_max × exp(-t × log(σ_max/σ_min))',
      description: 'Geometric progression of noise levels.',
      color: '#10b981',
      fn: (t) => {
        const sigmaMax = 1;
        const sigmaMin = 0.01;
        return sigmaMax * Math.exp(-t * Math.log(sigmaMax / sigmaMin));
      }
    }
  };

  // Draw schedule comparison
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

    const padding = { left: 50, right: 20, top: 30, bottom: 50 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(padding.left, padding.top);
    ctx.stroke();

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i / 10) * plotWidth;
      const y = height - padding.bottom - (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(x, height - padding.bottom);
      ctx.lineTo(x, padding.top);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '12px sans-serif';
    ctx.fillText('0', padding.left - 5, height - padding.bottom + 15);
    ctx.fillText('1', width - padding.right - 5, height - padding.bottom + 15);
    ctx.fillText('t (timestep)', width / 2 - 30, height - 10);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('σ (noise level)', 0, 0);
    ctx.restore();

    ctx.fillText('1', padding.left - 15, padding.top + 5);
    ctx.fillText('0', padding.left - 15, height - padding.bottom);

    // Draw all schedules (faded for non-selected)
    Object.entries(schedules).forEach(([key, schedule]) => {
      const isSelected = key === selectedSchedule;
      
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const t = i / 100;
        const sigma = schedule.fn(t);
        const x = padding.left + t * plotWidth;
        const y = height - padding.bottom - sigma * plotHeight;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = isSelected ? schedule.color : `${schedule.color}40`;
      ctx.lineWidth = isSelected ? 3 : 1.5;
      ctx.stroke();
    });

    // Draw discrete steps for selected schedule
    const schedule = schedules[selectedSchedule];
    ctx.fillStyle = schedule.color;
    for (let i = 0; i <= numSteps; i++) {
      const t = i / numSteps;
      const sigma = schedule.fn(t);
      const x = padding.left + t * plotWidth;
      const y = height - padding.bottom - sigma * plotHeight;
      
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Legend
    let legendY = padding.top + 10;
    Object.entries(schedules).forEach(([key, s]) => {
      const isSelected = key === selectedSchedule;
      ctx.fillStyle = s.color;
      ctx.fillRect(width - 120, legendY, 15, 3);
      ctx.fillStyle = isSelected ? 'white' : 'rgba(255, 255, 255, 0.5)';
      ctx.font = isSelected ? 'bold 11px sans-serif' : '11px sans-serif';
      ctx.fillText(s.name, width - 100, legendY + 4);
      legendY += 18;
    });

  }, [selectedSchedule, numSteps]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Sigma Schedules: <span className="text-fuchsia-600 dark:text-fuchsia-400">Noise Level Control</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How noise decreases during the generation process
        </p>
      </div>

      {/* Key Concept */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <div className="flex items-start gap-4">
          <Info className="text-fuchsia-600 dark:text-fuchsia-400 mt-1" size={24} />
          <div>
            <h3 className="font-bold text-lg text-fuchsia-300 mb-2">What is a Sigma Schedule?</h3>
            <p className="text-gray-700 dark:text-gray-300">
              The sigma schedule σ(t) defines how much noise is present at each timestep. 
              At t=0, we have maximum noise (σ ≈ 1). At t=1, we have minimum noise (σ ≈ 0).
              The <strong className="text-fuchsia-600 dark:text-fuchsia-400">shape of this curve</strong> affects 
              image quality and generation speed. Different schedules work better for 
              different scenarios.
            </p>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <canvas
          ref={canvasRef}
          width={700}
          height={350}
          className="w-full rounded-xl mb-4"
        />

        {/* Steps Control */}
        <div className="flex items-center gap-4 px-4">
          <span className="text-sm text-gray-800 dark:text-gray-400 w-32">Sampling Steps: {numSteps}</span>
          <input
            type="range"
            min="5"
            max="50"
            step="1"
            value={numSteps}
            onChange={(e) => setNumSteps(parseInt(e.target.value))}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* Schedule Selector */}
      <div className="grid md:grid-cols-2 gap-4">
        {Object.entries(schedules).map(([key, schedule]) => (
          <button
            key={key}
            onClick={() => setSelectedSchedule(key)}
            className={`p-4 rounded-xl border transition-all text-left ${
              selectedSchedule === key
                ? 'border-fuchsia-500 bg-fuchsia-500/20'
                : 'border-white/10 bg-black/30 hover:bg-white/5'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: schedule.color }} />
              <span className="font-bold">{schedule.name}</span>
            </div>
            <code className="text-xs text-gray-800 dark:text-gray-400 block mb-2">{schedule.formula}</code>
            <p className="text-sm text-gray-700 dark:text-gray-500">{schedule.description}</p>
          </button>
        ))}
      </div>

      {/* Schedule Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">When to Use Each Schedule</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Schedule</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Pros</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Cons</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Best For</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-fuchsia-600 dark:text-fuchsia-400">Linear</td>
                <td className="py-3 px-4">Simple, predictable</td>
                <td className="py-3 px-4">May miss details</td>
                <td className="py-3 px-4">Basic testing</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-purple-600 dark:text-purple-400">Cosine</td>
                <td className="py-3 px-4">Smooth transitions</td>
                <td className="py-3 px-4">Slower at extremes</td>
                <td className="py-3 px-4">High-res images</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">Karras</td>
                <td className="py-3 px-4">Optimized spacing</td>
                <td className="py-3 px-4">More complex</td>
                <td className="py-3 px-4">Production quality</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-green-400">Exponential</td>
                <td className="py-3 px-4">Perceptually uniform</td>
                <td className="py-3 px-4">Needs tuning</td>
                <td className="py-3 px-4">Custom workflows</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Relationship to SNR */}
      <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-2xl p-6 border border-purple-500/30">
        <h3 className="text-xl font-bold mb-4">Signal-to-Noise Ratio (SNR)</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-bold text-purple-300 mb-3">The Connection</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              The sigma schedule directly determines the SNR at each timestep:
            </p>
            <div className="bg-black/30 rounded-lg p-4 font-mono text-center">
              <p className="text-lg text-purple-300">SNR(t) = α²(t) / σ²(t)</p>
              <p className="text-sm text-gray-800 dark:text-gray-400 mt-2">where α(t) = √(1 - σ²(t))</p>
            </div>
          </div>
          <div>
            <h4 className="font-bold text-blue-300 mb-3">Why It Matters</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400">•</span>
                <span><strong>High SNR</strong> (low σ): Image is mostly visible, fine details</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400">•</span>
                <span><strong>Low SNR</strong> (high σ): Image is mostly noise, global structure</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 dark:text-blue-400">•</span>
                <span><strong>Karras insight:</strong> Equal log-SNR steps work best</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* SD3 Settings */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">SD3 Default Configuration</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/5 rounded-lg p-4">
            <h4 className="font-bold text-fuchsia-300 mb-2">Schedule</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">Linear or Karras (configurable)</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4">
            <h4 className="font-bold text-purple-300 mb-2">Steps</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">28-50 typical (quality/speed tradeoff)</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4">
            <h4 className="font-bold text-blue-300 mb-2">Sigma Range</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">σ_max=14.6, σ_min=0.0292</p>
          </div>
        </div>
      </div>
    </div>
  );
}
