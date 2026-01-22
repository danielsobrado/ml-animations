import React, { useState } from 'react';
import { ArrowRight, Info, Zap, Brain, Shuffle } from 'lucide-react';

export default function OverviewPanel() {
  const [hoveredComponent, setHoveredComponent] = useState(null);

  const components = {
    input: {
      title: 'Input Data (x)',
      description: 'Original data (images, text, etc.) that we want to learn to represent and reconstruct.',
      color: 'from-blue-500 to-cyan-500',
      details: ['Batch of samples', 'e.g., 28×28 images', 'Flattened or convolutional']
    },
    encoder: {
      title: 'Encoder q(z|x)',
      description: 'Neural network that maps input to parameters of the latent distribution (μ and σ).',
      color: 'from-green-500 to-emerald-500',
      details: ['Outputs μ (mean)', 'Outputs σ (std dev)', 'Learns to compress']
    },
    latent: {
      title: 'Latent Space z',
      description: 'Compressed probabilistic representation. We sample z ~ N(μ, σ²) using reparameterization.',
      color: 'from-purple-500 to-violet-500',
      details: ['Continuous & smooth', 'Reparameterization trick', 'z = μ + σ⊙ε']
    },
    decoder: {
      title: 'Decoder p(x|z)',
      description: 'Neural network that reconstructs data from latent samples.',
      color: 'from-orange-500 to-red-500',
      details: ['Generates output', 'Same shape as input', 'Learns to expand']
    },
    output: {
      title: 'Reconstruction (x̂)',
      description: 'Reconstructed output that should match the original input as closely as possible.',
      color: 'from-pink-500 to-rose-500',
      details: ['Same dims as input', 'Probabilistic output', 'Compare with x']
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          VAE Architecture: <span className="text-purple-600 dark:text-purple-400">The Big Picture</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          Autoencoders that learn probabilistic latent representations for generation
        </p>
      </div>

      {/* Main Architecture Diagram */}
      <div className="bg-black/30 rounded-2xl p-8 border border-white/10">
        <div className="flex items-center justify-between gap-4">
          {/* Input */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('input')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-24 h-24 rounded-xl bg-gradient-to-br ${components.input.color} flex items-center justify-center`}>
              <div className="grid grid-cols-4 gap-0.5">
                {Array(16).fill(0).map((_, i) => (
                  <div key={i} className={`w-3 h-3 rounded-sm ${Math.random() > 0.5 ? 'bg-white/80' : 'bg-white/30'}`} />
                ))}
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">Input x</p>
            <p className="text-center text-xs text-gray-700 dark:text-gray-500">[batch, 784]</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 flex-shrink-0" />

          {/* Encoder */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('encoder')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-32 h-32 rounded-xl bg-gradient-to-br ${components.encoder.color} flex flex-col items-center justify-center p-2`}>
              <Brain size={24} className="mb-1" />
              <p className="text-xs font-bold text-center">Encoder</p>
              <p className="text-xs opacity-80">q(z|x)</p>
              <div className="mt-2 flex gap-2">
                <div className="bg-white/20 rounded px-2 py-0.5 text-xs">μ</div>
                <div className="bg-white/20 rounded px-2 py-0.5 text-xs">σ</div>
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">Encoder</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 flex-shrink-0" />

          {/* Latent Space */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('latent')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-28 h-28 rounded-full bg-gradient-to-br ${components.latent.color} flex items-center justify-center relative`}>
              <div className="absolute inset-2 rounded-full gaussian-gradient latent-drift" />
              <div className="relative z-10 text-center">
                <p className="text-lg font-bold">z</p>
                <p className="text-xs opacity-80">~ N(μ, σ²)</p>
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">Latent Space</p>
            <p className="text-center text-xs text-gray-700 dark:text-gray-500">[batch, latent_dim]</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 flex-shrink-0" />

          {/* Decoder */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('decoder')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-32 h-32 rounded-xl bg-gradient-to-br ${components.decoder.color} flex flex-col items-center justify-center p-2`}>
              <Shuffle size={24} className="mb-1" />
              <p className="text-xs font-bold text-center">Decoder</p>
              <p className="text-xs opacity-80">p(x|z)</p>
            </div>
            <p className="text-center mt-2 text-sm font-medium">Decoder</p>
          </div>

          <ArrowRight className="text-gray-700 dark:text-gray-500 flex-shrink-0" />

          {/* Output */}
          <div
            className="relative cursor-pointer transition-transform hover:scale-105"
            onMouseEnter={() => setHoveredComponent('output')}
            onMouseLeave={() => setHoveredComponent(null)}
          >
            <div className={`w-24 h-24 rounded-xl bg-gradient-to-br ${components.output.color} flex items-center justify-center`}>
              <div className="grid grid-cols-4 gap-0.5">
                {Array(16).fill(0).map((_, i) => (
                  <div key={i} className={`w-3 h-3 rounded-sm ${Math.random() > 0.4 ? 'bg-white/70' : 'bg-white/30'}`} />
                ))}
              </div>
            </div>
            <p className="text-center mt-2 text-sm font-medium">Output x̂</p>
            <p className="text-center text-xs text-gray-700 dark:text-gray-500">[batch, 784]</p>
          </div>
        </div>

        {/* Hover Info Panel */}
        {hoveredComponent && (
          <div className="mt-6 p-4 bg-white/5 rounded-xl border border-white/10">
            <div className="flex items-start gap-3">
              <Info className="text-purple-600 dark:text-purple-400 mt-1" size={20} />
              <div>
                <h4 className="font-bold text-lg">{components[hoveredComponent].title}</h4>
                <p className="text-gray-700 dark:text-gray-300 mt-1">{components[hoveredComponent].description}</p>
                <div className="flex gap-2 mt-3">
                  {components[hoveredComponent].details.map((detail, i) => (
                    <span key={i} className="bg-white/10 px-2 py-1 rounded text-sm">{detail}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Key Concepts Grid */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-2 flex items-center gap-2">
            <Zap size={18} /> Why "Variational"?
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Unlike regular autoencoders, VAEs learn a <strong>probability distribution</strong> in 
            latent space, not just fixed points. This allows smooth interpolation and generation of new samples.
          </p>
        </div>

        <div className="bg-gradient-to-br from-green-900/30 to-green-800/20 rounded-xl p-5 border border-green-500/30">
          <h3 className="font-bold text-green-300 mb-2 flex items-center gap-2">
            <Brain size={18} /> The Reparameterization Trick
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            To backpropagate through sampling, we use: <code className="bg-black/30 px-1 rounded">z = μ + σ ⊙ ε</code> 
            where ε ~ N(0,1). This moves randomness outside the gradient path.
          </p>
        </div>

        <div className="bg-gradient-to-br from-orange-900/30 to-orange-800/20 rounded-xl p-5 border border-orange-500/30">
          <h3 className="font-bold text-orange-300 mb-2 flex items-center gap-2">
            <Shuffle size={18} /> Generation Power
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Once trained, we can generate new data by sampling z ~ N(0,1) and passing it through
            the decoder - no encoder needed! The learned latent space captures data structure.
          </p>
        </div>
      </div>

      {/* VAE vs AE Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">📊 VAE vs Regular Autoencoder</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-2 px-4">Feature</th>
                <th className="text-left py-2 px-4 text-blue-600 dark:text-blue-400">Regular Autoencoder</th>
                <th className="text-left py-2 px-4 text-purple-600 dark:text-purple-400">Variational AE</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4">Latent Space</td>
                <td className="py-2 px-4">Deterministic vectors</td>
                <td className="py-2 px-4">Probability distributions</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4">Encoder Output</td>
                <td className="py-2 px-4">z directly</td>
                <td className="py-2 px-4">μ and σ (then sample z)</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4">Loss Function</td>
                <td className="py-2 px-4">Reconstruction only</td>
                <td className="py-2 px-4">Reconstruction + KL Divergence</td>
              </tr>
              <tr className="border-b border-white/5">
                <td className="py-2 px-4">Generation</td>
                <td className="py-2 px-4">Poor (gaps in latent space)</td>
                <td className="py-2 px-4">Smooth (continuous latent space)</td>
              </tr>
              <tr>
                <td className="py-2 px-4">Interpolation</td>
                <td className="py-2 px-4">May produce artifacts</td>
                <td className="py-2 px-4">Meaningful interpolations</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Paper Reference */}
      <div className="bg-gradient-to-r from-purple-900/20 to-pink-900/20 rounded-xl p-4 border border-purple-500/20">
        <p className="text-sm text-gray-800 dark:text-gray-400">
          📄 <strong>Key Paper:</strong> "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013) - 
          Introduced VAEs with the reparameterization trick, enabling practical training of deep generative models.
        </p>
      </div>
    </div>
  );
}
