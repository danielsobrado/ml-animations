import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';
import gsap from 'gsap';

export default function AdaLNPanel() {
  const [timestep, setTimestep] = useState(500);
  const [isAnimating, setIsAnimating] = useState(false);
  const animRef = useRef(null);

  // Simulated conditioning outputs based on timestep
  const gamma = 1.0 + (timestep / 1000) * 0.5;  // scale
  const beta = (timestep / 1000) * 0.3 - 0.15;   // shift
  const alpha = Math.min(0.9, timestep / 1000);  // gate

  useEffect(() => {
    if (isAnimating) {
      animRef.current = gsap.to({}, {
        duration: 0.05,
        repeat: -1,
        onRepeat: () => {
          setTimestep(t => {
            const next = t - 5;
            return next < 0 ? 1000 : next;
          });
        },
      });
    } else if (animRef.current) {
      animRef.current.kill();
    }
    return () => { if (animRef.current) animRef.current.kill(); };
  }, [isAnimating]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-pink-600 dark:text-pink-400">AdaLN-Zero</span>: Adaptive Layer Normalization
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How DiT injects timestep and class conditioning into the model
        </p>
      </div>

      {/* Interactive Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => setIsAnimating(!isAnimating)}
            className="px-4 py-2 rounded-lg bg-pink-600 hover:bg-pink-500 flex items-center gap-2"
          >
            {isAnimating ? <Pause size={18} /> : <Play size={18} />}
            {isAnimating ? 'Pause' : 'Animate Denoising'}
          </button>
          <button
            onClick={() => { setTimestep(1000); setIsAnimating(false); }}
            className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 flex items-center gap-2"
          >
            <RotateCcw size={18} />
            Reset
          </button>
        </div>

        {/* Timestep Slider */}
        <div className="mb-8">
          <label className="text-sm text-gray-800 dark:text-gray-400 block mb-2">
            Timestep t: <span className="text-pink-600 dark:text-pink-400 font-bold">{timestep}</span> / 1000
          </label>
          <input
            type="range"
            min="0"
            max="1000"
            value={timestep}
            onChange={(e) => setTimestep(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-700 dark:text-gray-500 mt-1">
            <span>Clean (t=0)</span>
            <span>Noisy (t=1000)</span>
          </div>
        </div>

        {/* AdaLN Diagram */}
        <div className="bg-black/30 rounded-xl p-6">
          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            {/* Conditioning Input */}
            <div className="text-center">
              <div className="w-32 h-20 rounded-xl bg-gradient-to-br from-amber-600 to-orange-600 flex flex-col items-center justify-center mb-2">
                <span className="text-xs text-white/70">Conditioning</span>
                <span className="text-lg font-bold">c = t + y</span>
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400">timestep + class/text</p>
            </div>

            {/* MLP */}
            <div className="text-center">
              <div className="w-24 h-16 rounded-lg bg-gradient-to-br from-purple-600 to-violet-600 flex items-center justify-center mb-2">
                <span className="text-sm font-bold">MLP</span>
              </div>
              <p className="text-xs text-gray-800 dark:text-gray-400">Learn parameters</p>
            </div>

            {/* Parameters Output */}
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <div 
                  className="w-16 h-16 rounded-lg bg-gradient-to-br from-pink-600 to-rose-600 flex flex-col items-center justify-center transition-all"
                  style={{ transform: `scale(${0.8 + gamma * 0.2})` }}
                >
                  <span className="text-xs">γ (scale)</span>
                  <span className="text-lg font-bold">{gamma.toFixed(2)}</span>
                </div>
              </div>
              <div className="text-center">
                <div 
                  className="w-16 h-16 rounded-lg bg-gradient-to-br from-blue-600 to-cyan-600 flex flex-col items-center justify-center transition-all"
                  style={{ transform: `translateY(${beta * 20}px)` }}
                >
                  <span className="text-xs">β (shift)</span>
                  <span className="text-lg font-bold">{beta.toFixed(2)}</span>
                </div>
              </div>
              <div className="text-center">
                <div 
                  className="w-16 h-16 rounded-lg bg-gradient-to-br from-green-600 to-emerald-600 flex flex-col items-center justify-center transition-all"
                  style={{ opacity: 0.3 + alpha * 0.7 }}
                >
                  <span className="text-xs">α (gate)</span>
                  <span className="text-lg font-bold">{alpha.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Formula */}
          <div className="mt-8 text-center p-4 bg-white/5 rounded-xl">
            <p className="text-lg font-mono">
              AdaLN(x, c) = γ(c) × <span className="text-yellow-400">LayerNorm(x)</span> + β(c)
            </p>
            <p className="text-sm text-gray-800 dark:text-gray-400 mt-2">
              Output is then scaled by gate α: <span className="text-green-400">α(c)</span> × output
            </p>
          </div>
        </div>
      </div>

      {/* Why AdaLN */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-black/30 rounded-xl p-5 border border-white/10">
          <h3 className="font-bold text-pink-600 dark:text-pink-400 mb-3">🎯 Why Not Cross-Attention?</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            Traditional U-Net uses cross-attention to inject text. DiT instead:
          </p>
          <ul className="text-sm text-gray-800 dark:text-gray-400 space-y-1">
            <li>• Modulates <em>all</em> features, not just attended ones</li>
            <li>• More parameter efficient</li>
            <li>• Works well with scalar conditions (timestep, class)</li>
            <li>• Simpler architecture</li>
          </ul>
        </div>
        
        <div className="bg-black/30 rounded-xl p-5 border border-white/10">
          <h3 className="font-bold text-green-400 mb-3">🎚️ Why Zero Initialization?</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            The "Zero" in AdaLN-Zero refers to initializing gates α to 0:
          </p>
          <ul className="text-sm text-gray-800 dark:text-gray-400 space-y-1">
            <li>• At init, each block is an identity function</li>
            <li>• Gradients flow through residual connections</li>
            <li>• Model gradually learns to use each layer</li>
            <li>• More stable training for deep networks</li>
          </ul>
        </div>
      </div>

      {/* Conditioning Types */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">What Gets Injected via AdaLN?</h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-amber-900/20 rounded-xl p-4 border border-amber-500/30">
            <h4 className="font-bold text-amber-600 dark:text-amber-400 mb-2">⏱️ Timestep t</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              Sinusoidal embedding of the noise level. Critical for knowing how much to denoise.
            </p>
            <div className="mt-2 font-mono text-xs text-gray-700 dark:text-gray-500">
              t_emb = sin/cos(t * freqs)
            </div>
          </div>
          
          <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
            <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">🏷️ Class Label y</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              For class-conditional generation (ImageNet). Learned embedding per class.
            </p>
            <div className="mt-2 font-mono text-xs text-gray-700 dark:text-gray-500">
              y_emb = embed_table[class_id]
            </div>
          </div>
          
          <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">📝 Pooled Text (SD3)</h4>
            <p className="text-sm text-gray-800 dark:text-gray-400">
              CLIP pooled embedding provides global semantic context for text-to-image.
            </p>
            <div className="mt-2 font-mono text-xs text-gray-700 dark:text-gray-500">
              txt_pool = clip.encode(text)[0]
            </div>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">AdaLN Implementation</h3>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">{`class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, cond_size=None):
        super().__init__()
        cond_size = cond_size or hidden_size
        
        # LayerNorm (learnable affine disabled - we compute our own)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # MLP to produce gamma, beta, alpha from conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 3 * hidden_size)  # γ, β, α
        )
        
        # Zero-initialize the final layer
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        # c: (B, cond_size) conditioning embedding
        # x: (B, seq_len, hidden_size) features
        
        # Get gamma, beta, alpha from conditioning
        gamma, beta, alpha = self.adaLN_modulation(c).chunk(3, dim=-1)
        # Each: (B, hidden_size) -> broadcast to (B, 1, hidden_size)
        
        # Apply adaptive layer norm
        x_norm = self.norm(x)
        x_out = gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)
        
        return x_out, alpha  # Return alpha for gating outside`}</pre>
        </div>
      </div>

      {/* Visual Comparison */}
      <div className="bg-gradient-to-r from-pink-900/30 to-orange-900/30 rounded-xl p-6 border border-pink-500/30">
        <h3 className="font-bold text-pink-300 mb-4">AdaLN vs Standard LayerNorm</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <p className="text-sm text-gray-800 dark:text-gray-400 mb-2">Standard LayerNorm:</p>
            <div className="bg-black/30 p-3 rounded-lg font-mono text-sm">
              y = γ × norm(x) + β
            </div>
            <p className="text-xs text-gray-700 dark:text-gray-500 mt-2">γ, β are learned parameters (same for all inputs)</p>
          </div>
          <div>
            <p className="text-sm text-gray-800 dark:text-gray-400 mb-2">Adaptive LayerNorm:</p>
            <div className="bg-black/30 p-3 rounded-lg font-mono text-sm">
              y = <span className="text-pink-600 dark:text-pink-400">γ(c)</span> × norm(x) + <span className="text-blue-600 dark:text-blue-400">β(c)</span>
            </div>
            <p className="text-xs text-gray-700 dark:text-gray-500 mt-2">γ(c), β(c) are functions of conditioning c</p>
          </div>
        </div>
      </div>
    </div>
  );
}
