import React, { useState, useEffect } from 'react';
import { Scale, ChevronRight, BarChart3, Zap } from 'lucide-react';

export default function LossPanel() {
  const [beta, setBeta] = useState(1.0);
  const [reconLoss, setReconLoss] = useState(150);
  const [klLoss, setKlLoss] = useState(25);
  const [isAnimating, setIsAnimating] = useState(false);

  // Simulate training progress
  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setReconLoss(prev => Math.max(20, prev - Math.random() * 5));
        setKlLoss(prev => {
          const target = 10 + Math.random() * 5;
          return prev + (target - prev) * 0.1;
        });
      }, 200);
      return () => clearInterval(interval);
    }
  }, [isAnimating]);

  const totalLoss = reconLoss + beta * klLoss;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          VAE Loss: <span className="text-pink-600">The ELBO</span>
        </h2>
        <p className="text-gray-800">
          Evidence Lower Bound = Reconstruction Loss + KL Divergence
        </p>
      </div>

      {/* Loss Formula */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-center text-lg font-bold mb-4">The VAE Loss Function</h3>

        <div className="flex items-center justify-center gap-4 flex-wrap text-lg font-mono">
          <span className="text-gray-800">ℒ</span>
          <span className="text-gray-700">=</span>
          <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg px-4 py-2">
            <span className="text-blue-600">𝔼<sub>q(z|x)</sub>[log p(x|z)]</span>
          </div>
          <span className="text-gray-700">−</span>
          <div className="bg-purple-900/30 border border-purple-500/30 rounded-lg px-4 py-2">
            <span className="text-purple-600">β · KL(q(z|x) || p(z))</span>
          </div>
        </div>

        <div className="mt-4 grid md:grid-cols-2 gap-4">
          <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-600 flex items-center gap-2">
              <BarChart3 size={18} />
              Reconstruction Loss
            </h4>
            <p className="text-sm text-gray-700 mt-2">
              How well can we reconstruct the input from z? Usually Binary Cross-Entropy (for [0,1] pixels)
              or MSE. Encourages decoder to output x̂ ≈ x.
            </p>
            <p className="text-xs text-gray-700 mt-2 font-mono">
              BCE = -Σ[x·log(x̂) + (1-x)·log(1-x̂)]
            </p>
          </div>

          <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
            <h4 className="font-bold text-purple-600 flex items-center gap-2">
              <Scale size={18} />
              KL Divergence Loss
            </h4>
            <p className="text-sm text-gray-700 mt-2">
              Forces q(z|x) to be close to the prior p(z) = N(0, I). Regularizes the latent space
              to be smooth and continuous.
            </p>
            <p className="text-xs text-gray-700 mt-2 font-mono">
              KL = -½ Σ[1 + log(σ²) - μ² - σ²]
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Loss Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Zap size={20} className="text-yellow-400" />
          Interactive Loss Balance
        </h3>

        {/* Beta Slider */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium">β (KL Weight):</label>
            <span className="text-purple-600 font-mono">{beta.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="4"
            step="0.1"
            value={beta}
            onChange={(e) => setBeta(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-700 mt-1">
            <span>0 (No KL)</span>
            <span>1 (Standard VAE)</span>
            <span>4 (Strong regularization)</span>
          </div>
        </div>

        {/* Loss Bars */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-blue-600">Reconstruction Loss</span>
              <span className="font-mono">{reconLoss.toFixed(1)}</span>
            </div>
            <div className="h-6 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
                style={{ width: `${Math.min(100, reconLoss / 2)}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-purple-600">KL Loss × β</span>
              <span className="font-mono">{(beta * klLoss).toFixed(1)}</span>
            </div>
            <div className="h-6 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-purple-600 transition-all duration-300"
                style={{ width: `${Math.min(100, (beta * klLoss) / 2)}%` }}
              />
            </div>
          </div>

          <div className="pt-2 border-t border-white/10">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-pink-600 font-bold">Total Loss (ELBO)</span>
              <span className="font-mono font-bold">{totalLoss.toFixed(1)}</span>
            </div>
            <div className="h-8 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-pink-500 to-rose-600 transition-all duration-300"
                style={{ width: `${Math.min(100, totalLoss / 4)}%` }}
              />
            </div>
          </div>
        </div>

        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className="mt-4 px-4 py-2 bg-pink-600 hover:bg-pink-700 rounded-lg transition-colors"
        >
          {isAnimating ? 'Stop Training Simulation' : 'Simulate Training'}
        </button>
      </div>

      {/* β-VAE Explanation */}
      <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-2xl p-6 border border-purple-500/30">
        <h3 className="text-lg font-bold mb-3 text-purple-300">🔬 β-VAE: Disentanglement</h3>
        <p className="text-gray-700 mb-4">
          By increasing β {'>'}1, we get <strong>β-VAE</strong> which encourages more disentangled latent representations.
          Each latent dimension learns to capture a single factor of variation.
        </p>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-black/20 rounded-lg p-3">
            <p className="font-bold text-center">β = 0</p>
            <p className="text-xs text-gray-800 mt-1">
              Regular autoencoder. No KL constraint, no generation ability.
            </p>
          </div>
          <div className="bg-black/20 rounded-lg p-3 ring-2 ring-purple-500/50">
            <p className="font-bold text-center text-purple-600">β = 1</p>
            <p className="text-xs text-gray-800 mt-1">
              Standard VAE. Balance of reconstruction and regularization.
            </p>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <p className="font-bold text-center">β {'>'} 1</p>
            <p className="text-xs text-gray-800 mt-1">
              β-VAE. Stronger regularization → more disentanglement, blurrier reconstructions.
            </p>
          </div>
        </div>
      </div>

      {/* KL Divergence Deep Dive */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4">📐 KL Divergence Closed Form</h3>

        <p className="text-gray-700 mb-4">
          For Gaussian distributions, KL divergence has a closed-form solution (no sampling needed!):
        </p>

        <div className="bg-black/40 rounded-xl p-4 font-mono text-sm overflow-x-auto">
          <p className="text-purple-300">
            KL(q(z|x) || p(z)) = -½ Σ<sub>j=1</sub><sup>J</sup> (1 + log(σ<sub>j</sub>²) - μ<sub>j</sub>² - σ<sub>j</sub>²)
          </p>
          <p className="text-gray-700 mt-2">
            where J = latent_dim, and μ, σ are outputs from encoder
          </p>
        </div>

        <div className="mt-4 bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-300 mb-2">PyTorch Implementation:</h4>
          <pre className="text-sm overflow-x-auto">
            <code className="text-green-300">{`def kl_divergence(mu, logvar):
    # KL(q(z|x) || N(0, I))
    # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    # Reconstruction loss (BCE for binary images)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence
    kl_loss = kl_divergence(mu, logvar)

    return recon_loss + beta * kl_loss`}</code>
          </pre>
        </div>
      </div>

      {/* Trade-off Visualization */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
          <h4 className="font-bold text-blue-300 mb-2">⬆️ High Reconstruction, Low KL</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• Sharp, accurate reconstructions</li>
            <li>• Latent space may have "holes"</li>
            <li>• Poor generation quality</li>
            <li>• Overfitting to training data</li>
          </ul>
        </div>
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-300 mb-2">⬆️ Low Reconstruction, High KL</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• Blurry reconstructions</li>
            <li>• Smooth, continuous latent space</li>
            <li>• Better generation/interpolation</li>
            <li>• Information bottleneck</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
