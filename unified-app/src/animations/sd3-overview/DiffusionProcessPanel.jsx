import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Info } from 'lucide-react';

export default function DiffusionProcessPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(28);
  const canvasRef = useRef(null);

  // Animation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setStep(prev => {
        if (prev >= totalSteps) {
          setIsPlaying(false);
          return totalSteps;
        }
        return prev + 1;
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isPlaying, totalSteps]);

  // Canvas visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const t = step / totalSteps; // 0 to 1
    const sigma = 1 - t; // Noise level decreases

    // Draw noisy image representation
    const imageSize = 200;
    const imageX = (width - imageSize) / 2;
    const imageY = 30;

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(imageX, imageY, imageSize, imageSize);

    // Draw "image" with noise
    const gridSize = 10;
    for (let x = 0; x < imageSize; x += gridSize) {
      for (let y = 0; y < imageSize; y += gridSize) {
        // Base color (simulating an image - simple gradient for demo)
        const baseR = 100 + (x / imageSize) * 100;
        const baseG = 80 + (y / imageSize) * 120;
        const baseB = 150;

        // Add noise based on sigma
        const noise = (Math.random() - 0.5) * 200 * sigma;
        const r = Math.max(0, Math.min(255, baseR + noise));
        const g = Math.max(0, Math.min(255, baseG + noise));
        const b = Math.max(0, Math.min(255, baseB + noise));

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(imageX + x, imageY + y, gridSize, gridSize);
      }
    }

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.strokeRect(imageX, imageY, imageSize, imageSize);

    // Labels
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Step ${step} / ${totalSteps}`, width / 2, imageY + imageSize + 30);
    
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.fillText(`σ = ${sigma.toFixed(3)} (noise level)`, width / 2, imageY + imageSize + 50);
    ctx.fillText(`t = ${t.toFixed(3)} (time)`, width / 2, imageY + imageSize + 70);

    // Progress bar
    const barWidth = 300;
    const barX = (width - barWidth) / 2;
    const barY = imageY + imageSize + 90;
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(barX, barY, barWidth, 10);
    ctx.fillStyle = 'rgba(236, 72, 153, 0.8)';
    ctx.fillRect(barX, barY, barWidth * t, 10);

    // Noise to Data labels
    ctx.font = '11px sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'left';
    ctx.fillText('Noise', barX, barY + 25);
    ctx.textAlign = 'right';
    ctx.fillText('Data', barX + barWidth, barY + 25);

  }, [step, totalSteps]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Diffusion Process: <span className="text-fuchsia-600 dark:text-fuchsia-400">Noise to Image</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Watch the flow matching process transform noise into a coherent image
        </p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <canvas
          ref={canvasRef}
          width={500}
          height={380}
          className="w-full rounded-xl mb-4"
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
            onClick={() => { setStep(0); setIsPlaying(false); }}
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {/* Step Slider */}
        <div className="flex items-center gap-4 px-4">
          <span className="text-sm text-gray-800 dark:text-gray-400">Step</span>
          <input
            type="range"
            min="0"
            max={totalSteps}
            value={step}
            onChange={(e) => {
              setStep(parseInt(e.target.value));
              setIsPlaying(false);
            }}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-sm text-gray-800 dark:text-gray-400 w-16">{step}/{totalSteps}</span>
        </div>

        {/* Total Steps Control */}
        <div className="flex items-center gap-4 px-4 mt-4">
          <span className="text-sm text-gray-800 dark:text-gray-400">Total Steps</span>
          <input
            type="range"
            min="10"
            max="50"
            value={totalSteps}
            onChange={(e) => {
              setTotalSteps(parseInt(e.target.value));
              setStep(0);
            }}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-sm text-gray-800 dark:text-gray-400 w-16">{totalSteps}</span>
        </div>
      </div>

      {/* What Happens Each Step */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Info size={20} className="text-fuchsia-600 dark:text-fuchsia-400" />
          What Happens Each Step
        </h3>
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-fuchsia-600 dark:text-fuchsia-400 font-bold mb-2">1. Input</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Current noisy latent x_t + timestep t + text embeddings
            </p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-purple-600 dark:text-purple-400 font-bold mb-2">2. Predict</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              MMDiT predicts velocity v_θ(x_t, t, c) pointing toward data
            </p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-blue-600 dark:text-blue-400 font-bold mb-2">3. Update</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Euler step: x_{t+dt} = x_t + v × dt
            </p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-green-400 font-bold mb-2">4. Repeat</p>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Continue until t=1, noise → clean image
            </p>
          </div>
        </div>
      </div>

      {/* Classifier-Free Guidance */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Classifier-Free Guidance (CFG)</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              CFG improves prompt adherence by contrasting conditional and unconditional predictions:
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm text-center">
              <p className="text-fuchsia-300">v_guided = v_uncond + w × (v_cond - v_uncond)</p>
            </div>
            <p className="text-xs text-gray-700 dark:text-gray-500 mt-2">
              where w is the guidance scale (typically 4-8 for SD3)
            </p>
          </div>
          <div>
            <h4 className="font-bold text-purple-300 mb-2">Guidance Scale Effects</h4>
            <ul className="text-sm text-gray-800 dark:text-gray-400 space-y-2">
              <li><span className="text-blue-600 dark:text-blue-400">w = 1:</span> No guidance, may ignore prompt</li>
              <li><span className="text-green-400">w = 4-5:</span> Balanced, natural results</li>
              <li><span className="text-yellow-400">w = 7-8:</span> Strong prompt adherence</li>
              <li><span className="text-red-400">w &gt; 10:</span> Over-saturated, artifacts</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Training vs Inference */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Training vs Inference</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 rounded-xl p-4 border border-green-500/30">
            <h4 className="font-bold text-green-300 mb-3">Training</h4>
            <ol className="text-sm text-gray-800 dark:text-gray-400 space-y-2 list-decimal list-inside">
              <li>Sample image x₁ from dataset</li>
              <li>Encode to latent: z₁ = VAE.encode(x₁)</li>
              <li>Sample noise: z₀ ~ N(0, I)</li>
              <li>Sample timestep: t ~ LogitNormal(0, 1)</li>
              <li>Interpolate: z_t = (1-t)z₀ + tz₁</li>
              <li>Predict: v = MMDiT(z_t, t, text_emb)</li>
              <li>Loss: ||v - (z₁ - z₀)||²</li>
            </ol>
          </div>
          <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-300 mb-3">Inference</h4>
            <ol className="text-sm text-gray-800 dark:text-gray-400 space-y-2 list-decimal list-inside">
              <li>Encode prompt with text encoders</li>
              <li>Sample noise: z₀ ~ N(0, I)</li>
              <li>For t = 0 to 1 (N steps):</li>
              <li className="pl-4">a. Predict v = MMDiT(z_t, t, text_emb)</li>
              <li className="pl-4">b. Apply CFG: v = v_u + w(v_c - v_u)</li>
              <li className="pl-4">c. Euler step: z_{t+dt} = z_t + v × dt</li>
              <li>Decode: image = VAE.decode(z₁)</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
