import React, { useState, useEffect, useRef } from 'react';
import { Info, RefreshCw } from 'lucide-react';

export default function LogitNormalPanel() {
  const [mean, setMean] = useState(0);
  const [std, setStd] = useState(1);
  const [samples, setSamples] = useState([]);
  const canvasRef = useRef(null);
  const histCanvasRef = useRef(null);

  // Generate logit-normal samples
  const generateSamples = () => {
    const newSamples = [];
    for (let i = 0; i < 500; i++) {
      // Box-Muller transform for normal samples
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      
      // Transform: sigmoid(mean + std * z)
      const logit = mean + std * z;
      const t = 1 / (1 + Math.exp(-logit)); // sigmoid
      newSamples.push(t);
    }
    setSamples(newSamples);
  };

  useEffect(() => {
    generateSamples();
  }, [mean, std]);

  // Draw distribution curve
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

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, height - 40);
    ctx.lineTo(width - 20, height - 40);
    ctx.moveTo(40, height - 40);
    ctx.lineTo(40, 20);
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '12px sans-serif';
    ctx.fillText('0', 38, height - 20);
    ctx.fillText('0.5', width / 2 - 10, height - 20);
    ctx.fillText('1', width - 30, height - 20);
    ctx.fillText('t (timestep)', width / 2 - 30, height - 5);

    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('p(t)', 0, 0);
    ctx.restore();

    // Compute and draw PDF
    const logitNormalPDF = (t) => {
      if (t <= 0 || t >= 1) return 0;
      const logit = Math.log(t / (1 - t));
      const exponent = -((logit - mean) ** 2) / (2 * std ** 2);
      const normalization = 1 / (std * Math.sqrt(2 * Math.PI));
      const jacobian = 1 / (t * (1 - t));
      return normalization * Math.exp(exponent) * jacobian;
    };

    // Find max for scaling
    let maxPDF = 0;
    for (let i = 1; i < 100; i++) {
      const t = i / 100;
      maxPDF = Math.max(maxPDF, logitNormalPDF(t));
    }

    // Draw PDF curve
    ctx.beginPath();
    const plotWidth = width - 60;
    const plotHeight = height - 60;

    for (let i = 1; i <= 99; i++) {
      const t = i / 100;
      const pdf = logitNormalPDF(t);
      const x = 40 + (t * plotWidth);
      const y = height - 40 - (pdf / maxPDF) * plotHeight * 0.9;

      if (i === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.strokeStyle = '#ec4899';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo(40 + plotWidth, height - 40);
    ctx.lineTo(40, height - 40);
    ctx.closePath();
    ctx.fillStyle = 'rgba(236, 72, 153, 0.2)';
    ctx.fill();

    // Draw mean line
    const meanT = 1 / (1 + Math.exp(-mean));
    const meanX = 40 + (meanT * plotWidth);
    ctx.beginPath();
    ctx.moveTo(meanX, height - 40);
    ctx.lineTo(meanX, 30);
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#8b5cf6';
    ctx.fillText(`μ_t = ${meanT.toFixed(2)}`, meanX + 5, 25);

  }, [mean, std]);

  // Draw histogram of samples
  useEffect(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || samples.length === 0) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Create histogram bins
    const numBins = 30;
    const bins = new Array(numBins).fill(0);
    samples.forEach(s => {
      const bin = Math.min(Math.floor(s * numBins), numBins - 1);
      bins[bin]++;
    });

    const maxBin = Math.max(...bins);

    // Draw bars
    const barWidth = width / numBins;
    bins.forEach((count, i) => {
      const barHeight = (count / maxBin) * (height - 10);
      const x = i * barWidth;
      const y = height - barHeight;

      ctx.fillStyle = `rgba(236, 72, 153, ${0.4 + (count / maxBin) * 0.4})`;
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });

  }, [samples]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Logit-Normal Sampling: <span className="text-fuchsia-600 dark:text-fuchsia-400">Time Distribution</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How SD3 chooses which timesteps to train on
        </p>
      </div>

      {/* Key Concept */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <div className="flex items-start gap-4">
          <Info className="text-fuchsia-600 dark:text-fuchsia-400 mt-1" size={24} />
          <div>
            <h3 className="font-bold text-lg text-fuchsia-300 mb-2">Why Not Uniform Sampling?</h3>
            <p className="text-gray-700 dark:text-gray-300">
              During training, we sample random timesteps t to create training pairs. 
              <strong className="text-fuchsia-600 dark:text-fuchsia-400"> Uniform sampling</strong> treats all timesteps equally, 
              but not all timesteps are equally important! The middle timesteps (t ≈ 0.5) often 
              contain the most learnable signal. Logit-normal sampling lets us focus more 
              training on these critical timesteps.
            </p>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-lg font-bold mb-4">Logit-Normal Distribution</h3>
          <canvas
            ref={canvasRef}
            width={400}
            height={250}
            className="w-full rounded-lg mb-4"
          />

          {/* Controls */}
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
                Mean (μ): {mean.toFixed(1)} → mode at t ≈ {(1 / (1 + Math.exp(-mean))).toFixed(2)}
              </label>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={mean}
                onChange={(e) => setMean(parseFloat(e.target.value))}
                className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
                Std Dev (σ): {std.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.3"
                max="2"
                step="0.1"
                value={std}
                onChange={(e) => setStd(parseFloat(e.target.value))}
                className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold">Sampled Timesteps ({samples.length})</h3>
            <button
              onClick={generateSamples}
              className="flex items-center gap-2 px-3 py-1.5 bg-fuchsia-600 hover:bg-fuchsia-700 rounded-lg text-sm transition-colors"
            >
              <RefreshCw size={14} />
              Resample
            </button>
          </div>
          <canvas
            ref={histCanvasRef}
            width={400}
            height={200}
            className="w-full rounded-lg mb-4"
          />
          <p className="text-xs text-gray-700 dark:text-gray-500">
            Histogram of 500 samples from logit-normal distribution
          </p>
        </div>
      </div>

      {/* The Math */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">The Mathematics</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-bold text-fuchsia-300 mb-3">Sampling Process</h4>
            <div className="bg-black/30 rounded-lg p-4 font-mono text-sm space-y-2">
              <p className="text-gray-800 dark:text-gray-400"># 1. Sample from standard normal</p>
              <p><span className="text-blue-600 dark:text-blue-400">z</span> ~ N(0, 1)</p>
              <p className="text-gray-800 dark:text-gray-400 mt-2"># 2. Transform to logit scale</p>
              <p><span className="text-blue-600 dark:text-blue-400">logit</span> = μ + σ × z</p>
              <p className="text-gray-800 dark:text-gray-400 mt-2"># 3. Apply sigmoid</p>
              <p><span className="text-blue-600 dark:text-blue-400">t</span> = sigmoid(logit)</p>
              <p className="pl-4">= 1 / (1 + e^(-logit))</p>
            </div>
          </div>
          <div>
            <h4 className="font-bold text-purple-300 mb-3">PDF Formula</h4>
            <div className="bg-black/30 rounded-lg p-4 text-center">
              <p className="font-mono text-lg text-fuchsia-300 mb-3">
                p(t) = 1/(σ√(2π)) × 1/(t(1-t)) × e^(-(logit(t)-μ)²/(2σ²))
              </p>
              <p className="text-sm text-gray-800 dark:text-gray-400">
                where logit(t) = log(t/(1-t))
              </p>
            </div>
            <div className="mt-4 text-sm text-gray-800 dark:text-gray-400">
              <p>The key term <code className="bg-black/30 px-1 rounded">1/(t(1-t))</code> is the Jacobian 
              of the logit transform, which causes the distribution to concentrate away from t=0 and t=1.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Practical Values */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">SD3 Default Parameters</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-fuchsia-900/30 to-fuchsia-800/20 rounded-xl p-4 border border-fuchsia-500/30">
            <h4 className="font-bold text-fuchsia-300 mb-2">μ = 0.0</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Centers the distribution at t = 0.5, focusing training on the 
              "middle" of the diffusion process where denoising is most challenging.
            </p>
          </div>
          <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-4 border border-purple-500/30">
            <h4 className="font-bold text-purple-300 mb-2">σ = 1.0</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Controls spread. σ=1 gives good coverage while still concentrating 
              on important timesteps. Smaller σ = more focused.
            </p>
          </div>
          <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-300 mb-2">Why It Works</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Early timesteps (t≈0) have mostly noise. Late timesteps (t≈1) are 
              almost clean. The action is in the middle!
            </p>
          </div>
        </div>
      </div>

      {/* Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Sampling Strategies Compared</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Strategy</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Distribution</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Focus</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Used By</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">Uniform</td>
                <td className="py-3 px-4">t ~ U(0, 1)</td>
                <td className="py-3 px-4">Equal for all t</td>
                <td className="py-3 px-4">DDPM, DDIM</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-fuchsia-600 dark:text-fuchsia-400">Logit-Normal</td>
                <td className="py-3 px-4">t = σ(μ + σz)</td>
                <td className="py-3 px-4">Middle timesteps</td>
                <td className="py-3 px-4">SD3, Flux</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-purple-600 dark:text-purple-400">Mode Seeking</td>
                <td className="py-3 px-4">Shifted logit-normal</td>
                <td className="py-3 px-4">Near data (t≈1)</td>
                <td className="py-3 px-4">Fine-tuning</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
