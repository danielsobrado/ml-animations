import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Shuffle, Target } from 'lucide-react';

export default function LatentSpacePanel() {
  const [mode, setMode] = useState('gaussian');
  const [samples, setSamples] = useState([]);
  const [interpolationProgress, setInterpolationProgress] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const canvasRef = useRef(null);

  // Generate random samples in latent space
  const generateSamples = () => {
    const newSamples = [];
    for (let i = 0; i < 50; i++) {
      // Box-Muller transform for Gaussian
      const u1 = Math.random();
      const u2 = Math.random();
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      newSamples.push({ x: z1 * 0.8, y: z2 * 0.8, label: Math.floor(i / 5) });
    }
    setSamples(newSamples);
  };

  useEffect(() => {
    generateSamples();
  }, []);

  // Interpolation animation
  useEffect(() => {
    if (isAnimating && mode === 'interpolation') {
      const interval = setInterval(() => {
        setInterpolationProgress(prev => {
          if (prev >= 1) {
            return 0;
          }
          return prev + 0.02;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [isAnimating, mode]);

  const colors = [
    '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
    '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6'
  ];

  const drawDigit = (digit, size = 60) => {
    const patterns = {
      0: [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
      1: [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,1,0,0],[1,1,1,0]],
      2: [[1,1,1,0],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,1,1]],
      3: [[1,1,1,0],[0,0,1,0],[0,1,1,0],[0,0,1,0],[1,1,1,0]],
      4: [[1,0,1,0],[1,0,1,0],[1,1,1,1],[0,0,1,0],[0,0,1,0]],
      5: [[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,1,0],[1,1,1,0]],
      6: [[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0]],
      7: [[1,1,1,1],[0,0,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
      8: [[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]],
      9: [[0,1,1,0],[1,0,0,1],[0,1,1,1],[0,0,1,0],[0,1,0,0]],
    };
    const pattern = patterns[digit] || patterns[0];
    const cellSize = size / 5;
    
    return (
      <div className="grid" style={{ gridTemplateColumns: `repeat(4, ${cellSize}px)`, gap: '1px' }}>
        {pattern.flat().map((cell, i) => (
          <div 
            key={i} 
            className={`${cell ? 'bg-white' : 'bg-white/10'} rounded-sm`}
            style={{ width: cellSize, height: cellSize }}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Latent Space: <span className="text-purple-400">The Hidden World</span>
        </h2>
        <p className="text-gray-400">
          A continuous, structured space where similar inputs are close together
        </p>
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center gap-2">
        {[
          { id: 'gaussian', label: 'Gaussian Distribution', icon: Target },
          { id: 'clusters', label: 'Class Clusters', icon: Shuffle },
          { id: 'interpolation', label: 'Interpolation', icon: Play },
        ].map(m => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setIsAnimating(false); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              mode === m.id
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            <m.icon size={18} />
            {m.label}
          </button>
        ))}
      </div>

      {/* Visualization Area */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        {mode === 'gaussian' && (
          <div className="space-y-4">
            <h3 className="text-center text-lg font-bold text-purple-300">
              KL Divergence Forces Latent Space → N(0, I)
            </h3>
            <div className="relative h-80 bg-black/30 rounded-xl overflow-hidden">
              {/* Gaussian background */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-64 h-64 rounded-full gaussian-gradient opacity-50" />
                <div className="absolute w-48 h-48 rounded-full gaussian-gradient opacity-40" />
                <div className="absolute w-32 h-32 rounded-full gaussian-gradient opacity-30" />
              </div>
              
              {/* Axis labels */}
              <div className="absolute top-1/2 left-4 -translate-y-1/2 text-gray-500 text-sm">z₁</div>
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-gray-500 text-sm">z₂</div>
              
              {/* Sample points */}
              {samples.map((s, i) => (
                <div
                  key={i}
                  className="absolute w-3 h-3 rounded-full transition-all duration-500 latent-drift"
                  style={{
                    left: `calc(50% + ${s.x * 100}px)`,
                    top: `calc(50% + ${s.y * 100}px)`,
                    backgroundColor: colors[s.label % 10],
                    animationDelay: `${i * 50}ms`
                  }}
                />
              ))}
              
              {/* Origin cross */}
              <div className="absolute top-1/2 left-1/2 w-4 h-0.5 bg-gray-500 -translate-x-1/2" />
              <div className="absolute top-1/2 left-1/2 w-0.5 h-4 bg-gray-500 -translate-y-1/2" />
            </div>
            
            <button
              onClick={generateSamples}
              className="mx-auto flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg"
            >
              <Shuffle size={18} />
              Resample
            </button>
            
            <p className="text-center text-sm text-gray-400">
              The KL divergence loss encourages the learned distribution q(z|x) to be close to N(0, I),
              creating a smooth, continuous latent space.
            </p>
          </div>
        )}

        {mode === 'clusters' && (
          <div className="space-y-4">
            <h3 className="text-center text-lg font-bold text-purple-300">
              Similar Inputs → Nearby Latent Points
            </h3>
            <div className="relative h-80 bg-black/30 rounded-xl overflow-hidden">
              {/* Cluster regions */}
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(digit => {
                const angle = (digit / 10) * 2 * Math.PI;
                const radius = 80;
                return (
                  <div
                    key={digit}
                    className="absolute flex flex-col items-center"
                    style={{
                      left: `calc(50% + ${Math.cos(angle) * radius}px - 20px)`,
                      top: `calc(50% + ${Math.sin(angle) * radius}px - 20px)`,
                    }}
                  >
                    <div 
                      className="w-12 h-12 rounded-lg flex items-center justify-center opacity-80 border-2"
                      style={{ 
                        backgroundColor: colors[digit] + '40',
                        borderColor: colors[digit]
                      }}
                    >
                      <span className="font-bold">{digit}</span>
                    </div>
                    {/* Mini samples around each digit */}
                    {[0, 1, 2].map(j => (
                      <div
                        key={j}
                        className="absolute w-2 h-2 rounded-full"
                        style={{
                          backgroundColor: colors[digit],
                          left: `${20 + (Math.random() - 0.5) * 30}px`,
                          top: `${20 + (Math.random() - 0.5) * 30}px`,
                        }}
                      />
                    ))}
                  </div>
                );
              })}
              
              <div className="absolute top-4 left-4 text-xs text-gray-500">
                2D projection of latent space (t-SNE style)
              </div>
            </div>
            
            <p className="text-center text-sm text-gray-400">
              Different digit classes form distinct clusters in latent space.
              VAE learns meaningful features that separate similar from dissimilar inputs.
            </p>
          </div>
        )}

        {mode === 'interpolation' && (
          <div className="space-y-4">
            <h3 className="text-center text-lg font-bold text-purple-300">
              Smooth Interpolation in Latent Space
            </h3>
            
            <div className="flex items-center justify-center gap-8">
              {/* Start digit */}
              <div className="text-center">
                <div className="bg-blue-900/30 p-4 rounded-xl border border-blue-500/30">
                  {drawDigit(3, 50)}
                </div>
                <p className="text-sm text-blue-400 mt-2">z_start (digit 3)</p>
              </div>

              {/* Interpolation path */}
              <div className="flex-1 max-w-md">
                <div className="relative h-16 bg-gradient-to-r from-blue-600/30 via-purple-600/30 to-orange-600/30 rounded-full">
                  {/* Progress indicator */}
                  <div 
                    className="absolute top-1/2 -translate-y-1/2 w-6 h-6 bg-white rounded-full shadow-lg transition-all duration-100"
                    style={{ left: `calc(${interpolationProgress * 100}% - 12px)` }}
                  />
                  {/* Formula */}
                  <div className="absolute -bottom-8 left-0 right-0 text-center text-xs text-gray-400 font-mono">
                    z_interp = (1-t) × z_start + t × z_end, t = {interpolationProgress.toFixed(2)}
                  </div>
                </div>
              </div>

              {/* End digit */}
              <div className="text-center">
                <div className="bg-orange-900/30 p-4 rounded-xl border border-orange-500/30">
                  {drawDigit(8, 50)}
                </div>
                <p className="text-sm text-orange-400 mt-2">z_end (digit 8)</p>
              </div>
            </div>

            {/* Interpolated outputs */}
            <div className="mt-12 pt-4">
              <p className="text-center text-sm text-gray-400 mb-4">Generated outputs along the interpolation path:</p>
              <div className="flex justify-center gap-2">
                {[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1].map((t, i) => (
                  <div 
                    key={i} 
                    className={`p-2 rounded-lg transition-all ${
                      Math.abs(t - interpolationProgress) < 0.1 ? 'bg-purple-600/50 scale-110' : 'bg-white/5'
                    }`}
                  >
                    {drawDigit(Math.round(3 + (8 - 3) * t) % 10, 35)}
                    <p className="text-center text-xs text-gray-500 mt-1">{t.toFixed(2)}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-center gap-3 mt-4">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg"
              >
                {isAnimating ? <Pause size={18} /> : <Play size={18} />}
                {isAnimating ? 'Pause' : 'Animate'}
              </button>
              <button
                onClick={() => setInterpolationProgress(0)}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg"
              >
                <RotateCcw size={18} />
                Reset
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Key Insights */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
          <h4 className="font-bold text-blue-300 mb-2">🎯 Continuity</h4>
          <p className="text-sm text-gray-300">
            Small changes in z produce small changes in output. No "dead zones" in latent space thanks to KL regularization.
          </p>
        </div>
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-300 mb-2">✨ Generation</h4>
          <p className="text-sm text-gray-300">
            Sample z ~ N(0, I) and decode to generate new data. The decoder has learned to map any point in this space.
          </p>
        </div>
        <div className="bg-pink-900/20 rounded-xl p-4 border border-pink-500/30">
          <h4 className="font-bold text-pink-300 mb-2">🔀 Interpolation</h4>
          <p className="text-sm text-gray-300">
            Linear interpolation between two latent codes produces meaningful transitions between their decoded outputs.
          </p>
        </div>
      </div>

      {/* Dimensionality Info */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold mb-3">📏 Latent Dimension Choices</h4>
        <div className="grid grid-cols-4 gap-4 text-center">
          {[
            { dim: 2, use: 'Visualization', notes: 'Can plot, but limited capacity' },
            { dim: 20, use: 'MNIST digits', notes: 'Good balance for simple images' },
            { dim: 128, use: 'Face images', notes: 'More capacity for complex data' },
            { dim: 512, use: 'High-res images', notes: 'Even more expressive power' },
          ].map((d, i) => (
            <div key={i} className="bg-white/5 rounded-lg p-3">
              <p className="text-2xl font-bold text-purple-400">{d.dim}</p>
              <p className="text-sm font-medium">{d.use}</p>
              <p className="text-xs text-gray-500">{d.notes}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
