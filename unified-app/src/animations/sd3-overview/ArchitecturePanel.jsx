import React, { useState } from 'react';
import { Info, ArrowRight, Zap, Brain, Image, FileText, Layers } from 'lucide-react';

export default function ArchitecturePanel() {
  const [hoveredComponent, setHoveredComponent] = useState(null);
  const [selectedFlow, setSelectedFlow] = useState('training');

  const components = {
    prompt: {
      title: 'Text Prompt',
      description: 'User input describing the desired image. This drives the entire generation process.',
      color: 'from-blue-500 to-cyan-500',
      details: ['Natural language', 'Descriptive text', 'Style instructions']
    },
    clip: {
      title: 'CLIP Text Encoder',
      description: 'Encodes text into embeddings that understand image-text relationships.',
      color: 'from-green-500 to-emerald-500',
      details: ['77 tokens max', '768/1024 dim', 'Pooled output']
    },
    t5: {
      title: 'T5 Text Encoder',
      description: 'Provides rich, detailed text understanding with longer context.',
      color: 'from-teal-500 to-cyan-500',
      details: ['512 tokens', '4096 dim', 'Better semantics']
    },
    vae_enc: {
      title: 'VAE Encoder',
      description: 'Compresses input image to latent space (training only).',
      color: 'from-purple-500 to-violet-500',
      details: ['8x compression', 'Latent space', '16 channels']
    },
    mmdit: {
      title: 'MMDiT (Transformer)',
      description: 'The core diffusion model. Jointly attends to text and image tokens.',
      color: 'from-fuchsia-500 to-pink-500',
      details: ['Joint attention', 'AdaLN-Zero', '38 layers']
    },
    scheduler: {
      title: 'Flow Matching Scheduler',
      description: 'Controls the denoising process using flow-based sampling.',
      color: 'from-orange-500 to-amber-500',
      details: ['Euler/Heun', 'Logit-normal', '28-50 steps']
    },
    vae_dec: {
      title: 'VAE Decoder',
      description: 'Converts latent representation back to pixel space.',
      color: 'from-rose-500 to-red-500',
      details: ['Upsampling', 'Final image', '1024x1024']
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          SD3 Architecture: <span className="text-fuchsia-600 dark:text-fuchsia-400">The Complete System</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How text becomes images through diffusion
        </p>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-3xl font-bold text-fuchsia-600 dark:text-fuchsia-400">8B</p>
          <p className="text-sm text-gray-800 dark:text-gray-400">Parameters (largest)</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">3</p>
          <p className="text-sm text-gray-800 dark:text-gray-400">Text Encoders</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">1024²</p>
          <p className="text-sm text-gray-800 dark:text-gray-400">Max Resolution</p>
        </div>
        <div className="bg-black/30 rounded-xl p-4 text-center border border-white/10">
          <p className="text-3xl font-bold text-green-400">28-50</p>
          <p className="text-sm text-gray-800 dark:text-gray-400">Typical Steps</p>
        </div>
      </div>

      {/* Flow Selection */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setSelectedFlow('training')}
          className={`px-6 py-2 rounded-lg transition-all ${
            selectedFlow === 'training' ? 'bg-fuchsia-600' : 'bg-white/10 hover:bg-white/20'
          }`}
        >
          Training Flow
        </button>
        <button
          onClick={() => setSelectedFlow('inference')}
          className={`px-6 py-2 rounded-lg transition-all ${
            selectedFlow === 'inference' ? 'bg-fuchsia-600' : 'bg-white/10 hover:bg-white/20'
          }`}
        >
          Inference Flow
        </button>
      </div>

      {/* Architecture Diagram */}
      <div className="bg-black/30 rounded-2xl p-8 border border-white/10">
        {selectedFlow === 'inference' ? (
          // Inference Flow
          <div className="space-y-6">
            {/* Text Encoding Row */}
            <div className="flex items-center justify-center gap-4 flex-wrap">
              <ComponentBox
                id="prompt"
                data={components.prompt}
                hoveredComponent={hoveredComponent}
                setHoveredComponent={setHoveredComponent}
                icon={<FileText size={20} />}
              />
              <ArrowRight className="text-gray-700 dark:text-gray-500" />
              <div className="flex flex-col gap-2">
                <ComponentBox
                  id="clip"
                  data={components.clip}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<Brain size={20} />}
                  small
                />
                <ComponentBox
                  id="t5"
                  data={components.t5}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<Brain size={20} />}
                  small
                />
              </div>
            </div>

            {/* Arrows down */}
            <div className="flex justify-center">
              <div className="h-8 w-px bg-gradient-to-b from-green-500 to-fuchsia-500" />
            </div>

            {/* Diffusion Loop */}
            <div className="flex items-center justify-center gap-4">
              <div className="text-center">
                <div className="w-20 h-20 rounded-lg bg-gray-700 flex items-center justify-center mb-2">
                  <p className="text-2xl">🎲</p>
                </div>
                <p className="text-xs text-gray-800 dark:text-gray-400">Random Noise</p>
              </div>
              <ArrowRight className="text-gray-700 dark:text-gray-500" />
              <ComponentBox
                id="mmdit"
                data={components.mmdit}
                hoveredComponent={hoveredComponent}
                setHoveredComponent={setHoveredComponent}
                icon={<Layers size={20} />}
                large
              />
              <div className="flex items-center gap-2">
                <ArrowRight className="text-gray-700 dark:text-gray-500" />
                <ComponentBox
                  id="scheduler"
                  data={components.scheduler}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<Zap size={20} />}
                />
              </div>
            </div>

            {/* Loop indicator */}
            <div className="flex justify-center">
              <div className="bg-fuchsia-500/20 border border-fuchsia-500/50 rounded-lg px-4 py-2 text-sm">
                ↻ Repeat for N steps (velocity → denoise → repeat)
              </div>
            </div>

            {/* VAE Decode */}
            <div className="flex items-center justify-center gap-4">
              <div className="text-center">
                <div className="w-16 h-16 rounded-lg bg-purple-600/30 flex items-center justify-center mb-2 border border-purple-500/50">
                  <p className="text-xs">Latent</p>
                </div>
                <p className="text-xs text-gray-800 dark:text-gray-400">128×128×16</p>
              </div>
              <ArrowRight className="text-gray-700 dark:text-gray-500" />
              <ComponentBox
                id="vae_dec"
                data={components.vae_dec}
                hoveredComponent={hoveredComponent}
                setHoveredComponent={setHoveredComponent}
                icon={<Image size={20} />}
              />
              <ArrowRight className="text-gray-700 dark:text-gray-500" />
              <div className="text-center">
                <div className="w-24 h-24 rounded-lg bg-gradient-to-br from-rose-500/30 to-orange-500/30 flex items-center justify-center mb-2 border border-rose-500/50">
                  <p className="text-2xl">🖼️</p>
                </div>
                <p className="text-xs text-gray-800 dark:text-gray-400">1024×1024 RGB</p>
              </div>
            </div>
          </div>
        ) : (
          // Training Flow
          <div className="space-y-6">
            {/* Input Row */}
            <div className="flex items-center justify-center gap-4 flex-wrap">
              <div className="flex flex-col gap-2 items-center">
                <ComponentBox
                  id="prompt"
                  data={components.prompt}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<FileText size={20} />}
                />
                <p className="text-xs text-gray-700 dark:text-gray-500">Caption</p>
              </div>
              <div className="flex flex-col gap-2 items-center">
                <div className="w-20 h-20 rounded-lg bg-gradient-to-br from-rose-500/30 to-orange-500/30 flex items-center justify-center border border-rose-500/50">
                  <p className="text-2xl">🖼️</p>
                </div>
                <p className="text-xs text-gray-700 dark:text-gray-500">Target Image</p>
              </div>
            </div>

            {/* Processing Row */}
            <div className="flex items-center justify-center gap-4">
              <div className="flex flex-col gap-2">
                <ComponentBox
                  id="clip"
                  data={components.clip}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<Brain size={20} />}
                  small
                />
                <ComponentBox
                  id="t5"
                  data={components.t5}
                  hoveredComponent={hoveredComponent}
                  setHoveredComponent={setHoveredComponent}
                  icon={<Brain size={20} />}
                  small
                />
              </div>
              <ComponentBox
                id="vae_enc"
                data={components.vae_enc}
                hoveredComponent={hoveredComponent}
                setHoveredComponent={setHoveredComponent}
                icon={<Layers size={20} />}
              />
            </div>

            {/* Training objective */}
            <div className="flex items-center justify-center gap-4">
              <div className="bg-white/5 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-800 dark:text-gray-400">Text Embeddings</p>
              </div>
              <div className="bg-white/5 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-800 dark:text-gray-400">x₁ (clean latent)</p>
              </div>
              <div className="bg-white/5 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-800 dark:text-gray-400">x₀ ~ N(0,I)</p>
              </div>
            </div>

            <div className="flex justify-center">
              <div className="h-8 w-px bg-gradient-to-b from-purple-500 to-fuchsia-500" />
            </div>

            <div className="flex items-center justify-center gap-4">
              <ComponentBox
                id="mmdit"
                data={components.mmdit}
                hoveredComponent={hoveredComponent}
                setHoveredComponent={setHoveredComponent}
                icon={<Layers size={20} />}
                large
              />
            </div>

            {/* Loss */}
            <div className="flex justify-center">
              <div className="bg-gradient-to-r from-fuchsia-500/20 to-purple-500/20 border border-fuchsia-500/50 rounded-lg px-6 py-3">
                <p className="font-mono text-center">
                  Loss = ||v_θ(x_t, t, c) - (x₁ - x₀)||²
                </p>
                <p className="text-xs text-gray-800 dark:text-center mt-1">
                  Predict velocity from noisy input conditioned on text
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Hover Info */}
        {hoveredComponent && (
          <div className="mt-6 p-4 bg-white/5 rounded-xl border border-white/10">
            <div className="flex items-start gap-3">
              <Info className="text-fuchsia-600 dark:text-fuchsia-400 mt-1" size={20} />
              <div>
                <h4 className="font-bold text-lg">{components[hoveredComponent].title}</h4>
                <p className="text-gray-700 dark:text-gray-300 mt-1">{components[hoveredComponent].description}</p>
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

      {/* Key Innovations */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-br from-fuchsia-900/30 to-fuchsia-800/20 rounded-xl p-5 border border-fuchsia-500/30">
          <h3 className="font-bold text-fuchsia-300 mb-2 flex items-center gap-2">
            <Zap size={18} /> Multi-Modal DiT (MMDiT)
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Unlike U-Net based models, SD3 uses a Transformer that jointly processes 
            image and text tokens with bidirectional attention.
          </p>
        </div>

        <div className="bg-gradient-to-br from-purple-900/30 to-purple-800/20 rounded-xl p-5 border border-purple-500/30">
          <h3 className="font-bold text-purple-300 mb-2 flex items-center gap-2">
            <Brain size={18} /> Triple Text Encoders
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            CLIP-L, CLIP-G, and T5-XXL provide complementary text understanding:
            visual-language alignment + rich semantic comprehension.
          </p>
        </div>

        <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-xl p-5 border border-blue-500/30">
          <h3 className="font-bold text-blue-300 mb-2 flex items-center gap-2">
            <Layers size={18} /> Flow Matching
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Replaces score-based diffusion with simpler flow-based training,
            enabling better noise schedules and faster sampling.
          </p>
        </div>
      </div>
    </div>
  );
}

function ComponentBox({ id, data, hoveredComponent, setHoveredComponent, icon, small, large }) {
  const sizeClasses = large 
    ? 'w-32 h-32' 
    : small 
    ? 'w-24 h-16' 
    : 'w-28 h-20';

  return (
    <div
      className={`${sizeClasses} rounded-xl bg-gradient-to-br ${data.color} flex flex-col items-center justify-center cursor-pointer transition-all ${
        hoveredComponent === id ? 'scale-110 shadow-lg' : 'hover:scale-105'
      }`}
      onMouseEnter={() => setHoveredComponent(id)}
      onMouseLeave={() => setHoveredComponent(null)}
    >
      {icon}
      <p className={`${small ? 'text-xs' : 'text-sm'} font-bold text-center mt-1`}>
        {data.title.split(' ')[0]}
      </p>
    </div>
  );
}
