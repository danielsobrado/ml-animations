import React, { useState } from 'react';
import { Layers, ArrowRight } from 'lucide-react';

export default function ConcatPanel() {
  const [imageTokens, setImageTokens] = useState(16);
  const [textTokens, setTextTokens] = useState(8);

  const imgTokenArray = [...Array(Math.min(imageTokens, 16))];
  const txtTokenArray = [...Array(Math.min(textTokens, 12))];

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Token <span className="text-violet-400">Concatenation</span>
        </h2>
        <p className="text-gray-800 dark:text-gray-400">
          How image patches and text tokens become a unified sequence
        </p>
      </div>

      {/* Interactive Token Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Build Your Sequence</h3>
        
        {/* Controls */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="text-sm text-blue-600 dark:text-blue-400 block mb-2">
              Image Tokens: {imageTokens} (from {Math.sqrt(imageTokens)}×{Math.sqrt(imageTokens)} patches)
            </label>
            <input
              type="range"
              min="4"
              max="64"
              step="4"
              value={imageTokens}
              onChange={(e) => setImageTokens(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="text-sm text-orange-600 dark:text-orange-400 block mb-2">
              Text Tokens: {textTokens}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={textTokens}
              onChange={(e) => setTextTokens(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Token Visualization */}
        <div className="bg-black/30 rounded-xl p-6">
          {/* Before - Separate */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-8 mb-8">
            {/* Image tokens */}
            <div className="text-center">
              <div className={`grid gap-1 mb-2`} style={{ gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(imageTokens))}, minmax(0, 1fr))` }}>
                {imgTokenArray.map((_, i) => (
                  <div
                    key={i}
                    className="w-6 h-6 md:w-8 md:h-8 rounded bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-xs font-bold"
                  >
                    {i}
                  </div>
                ))}
              </div>
              <p className="text-blue-600 dark:text-sm">Image Tokens</p>
            </div>

            <div className="text-gray-700 dark:text-2xl">+</div>

            {/* Text tokens */}
            <div className="text-center">
              <div className="flex flex-wrap gap-1 mb-2 justify-center max-w-xs">
                {txtTokenArray.map((_, i) => (
                  <div
                    key={i}
                    className="px-2 py-1 rounded bg-gradient-to-br from-orange-500 to-amber-500 text-xs font-bold"
                  >
                    T{i}
                  </div>
                ))}
              </div>
              <p className="text-orange-600 dark:text-sm">Text Tokens</p>
            </div>
          </div>

          <ArrowRight className="mx-auto text-violet-400 mb-4" size={32} />

          {/* After - Concatenated */}
          <div className="text-center">
            <p className="text-sm text-gray-800 dark:text-gray-400 mb-3">Concatenated Sequence (length: {imageTokens + textTokens})</p>
            <div className="flex flex-wrap gap-1 justify-center p-4 bg-violet-900/20 rounded-xl border border-violet-500/30">
              {imgTokenArray.map((_, i) => (
                <div
                  key={`img-${i}`}
                  className="w-6 h-6 rounded bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-xs font-bold"
                >
                  {i}
                </div>
              ))}
              <div className="w-px h-6 bg-white/30 mx-1" />
              {txtTokenArray.map((_, i) => (
                <div
                  key={`txt-${i}`}
                  className="px-2 h-6 rounded bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center text-xs font-bold"
                >
                  T{i}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Token Embedding Details */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Layers size={20} className="text-violet-400" />
          Embedding Process
        </h3>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Image Embedding */}
          <div className="bg-blue-900/20 rounded-xl p-5 border border-blue-500/30">
            <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-3">Image Tokens</h4>
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-blue-500/30 flex items-center justify-center">1</span>
                <span>VAE encodes image → 128×128×16 latent</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-blue-500/30 flex items-center justify-center">2</span>
                <span>Patchify into 2×2 patches → 64×64 patches</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-blue-500/30 flex items-center justify-center">3</span>
                <span>Linear projection → d_model (1536/3072)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-blue-500/30 flex items-center justify-center">4</span>
                <span>Add 2D positional encoding</span>
              </div>
            </div>
          </div>

          {/* Text Embedding */}
          <div className="bg-orange-900/20 rounded-xl p-5 border border-orange-500/30">
            <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">Text Tokens</h4>
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-orange-500/30 flex items-center justify-center">1</span>
                <span>Tokenize prompt with CLIP/T5</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-orange-500/30 flex items-center justify-center">2</span>
                <span>Encode through CLIP-L, CLIP-G, T5</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-orange-500/30 flex items-center justify-center">3</span>
                <span>Concatenate encoder outputs</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-8 h-8 rounded bg-orange-500/30 flex items-center justify-center">4</span>
                <span>Project to match d_model</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dimension Breakdown */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Dimension Breakdown (SD3-Medium)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Stage</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Shape</th>
                <th className="py-3 px-4 text-left text-gray-800 dark:text-gray-400">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">Input Image</td>
                <td className="py-3 px-4 font-mono">1024×1024×3</td>
                <td className="py-3 px-4">RGB image</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">VAE Latent</td>
                <td className="py-3 px-4 font-mono">128×128×16</td>
                <td className="py-3 px-4">8× compression</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">Image Tokens</td>
                <td className="py-3 px-4 font-mono">4096×1536</td>
                <td className="py-3 px-4">64×64 patches × d_model</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-orange-600 dark:text-orange-400">Text (CLIP-L)</td>
                <td className="py-3 px-4 font-mono">77×768</td>
                <td className="py-3 px-4">CLIP ViT-L</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-orange-600 dark:text-orange-400">Text (CLIP-G)</td>
                <td className="py-3 px-4 font-mono">77×1280</td>
                <td className="py-3 px-4">CLIP ViT-bigG</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-orange-600 dark:text-orange-400">Text (T5)</td>
                <td className="py-3 px-4 font-mono">256×4096</td>
                <td className="py-3 px-4">T5-XXL encoder</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-violet-400">Combined Text</td>
                <td className="py-3 px-4 font-mono">~154×1536</td>
                <td className="py-3 px-4">Projected & concat</td>
              </tr>
              <tr className="bg-violet-900/20">
                <td className="py-3 px-4 text-violet-400 font-bold">Joint Sequence</td>
                <td className="py-3 px-4 font-mono font-bold">~4250×1536</td>
                <td className="py-3 px-4">Ready for attention</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Concatenation in Code</h3>
        <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <pre className="text-gray-700 dark:text-gray-300">{`# Pseudo-code for token concatenation
def prepare_joint_sequence(image_latent, text_embeddings):
    # Image tokens: (B, H, W, C) -> (B, H*W, D)
    img_tokens = patchify_and_embed(image_latent)  # (B, 4096, 1536)
    img_tokens = img_tokens + pos_embed_2d         # Add 2D position
    
    # Text tokens: already (B, seq_len, D)
    txt_tokens = project_text(text_embeddings)     # (B, ~154, 1536)
    
    # Concatenate along sequence dimension
    joint_seq = torch.cat([img_tokens, txt_tokens], dim=1)
    # joint_seq shape: (B, 4096 + 154, 1536) = (B, 4250, 1536)
    
    return joint_seq, len(img_tokens[0])  # Also return split point`}</pre>
        </div>
      </div>
    </div>
  );
}
