import React, { useState, useEffect, useRef } from 'react';
import { Info, ArrowRight, ZoomIn, ZoomOut } from 'lucide-react';

export default function LatentSpacePanel() {
  const [compressionRatio, setCompressionRatio] = useState(8);
  const canvasRef = useRef(null);

  // Draw latent space visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw pixel space (left)
    const pixelSize = 150;
    const pixelX = 50;
    const pixelY = (height - pixelSize) / 2;

    ctx.fillStyle = 'rgba(236, 72, 153, 0.2)';
    ctx.fillRect(pixelX, pixelY, pixelSize, pixelSize);
    ctx.strokeStyle = 'rgba(236, 72, 153, 0.8)';
    ctx.lineWidth = 2;
    ctx.strokeRect(pixelX, pixelY, pixelSize, pixelSize);

    // Draw pixel grid
    ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)';
    ctx.lineWidth = 1;
    const pixelGridSize = pixelSize / 8;
    for (let i = 1; i < 8; i++) {
      ctx.beginPath();
      ctx.moveTo(pixelX + i * pixelGridSize, pixelY);
      ctx.lineTo(pixelX + i * pixelGridSize, pixelY + pixelSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pixelX, pixelY + i * pixelGridSize);
      ctx.lineTo(pixelX + pixelSize, pixelY + i * pixelGridSize);
      ctx.stroke();
    }

    // Label
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Pixel Space', pixelX + 35, pixelY - 10);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fillText('1024 × 1024 × 3', pixelX + 25, pixelY + pixelSize + 20);
    ctx.fillText('= 3.1M values', pixelX + 35, pixelY + pixelSize + 35);

    // Arrow
    ctx.beginPath();
    ctx.moveTo(pixelX + pixelSize + 30, height / 2);
    ctx.lineTo(pixelX + pixelSize + 80, height / 2);
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pixelX + pixelSize + 80, height / 2);
    ctx.lineTo(pixelX + pixelSize + 70, height / 2 - 8);
    ctx.lineTo(pixelX + pixelSize + 70, height / 2 + 8);
    ctx.closePath();
    ctx.fillStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.fill();

    // VAE Encoder
    ctx.fillStyle = 'rgba(139, 92, 246, 0.3)';
    ctx.fillRect(pixelX + pixelSize + 90, height / 2 - 40, 80, 80);
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.strokeRect(pixelX + pixelSize + 90, height / 2 - 40, 80, 80);
    ctx.fillStyle = 'white';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText('VAE', pixelX + pixelSize + 115, height / 2 - 5);
    ctx.fillText('Encoder', pixelX + pixelSize + 103, height / 2 + 12);

    // Arrow 2
    ctx.beginPath();
    ctx.moveTo(pixelX + pixelSize + 180, height / 2);
    ctx.lineTo(pixelX + pixelSize + 230, height / 2);
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pixelX + pixelSize + 230, height / 2);
    ctx.lineTo(pixelX + pixelSize + 220, height / 2 - 8);
    ctx.lineTo(pixelX + pixelSize + 220, height / 2 + 8);
    ctx.closePath();
    ctx.fill();

    // Latent space (right)
    const latentSize = pixelSize / compressionRatio * 2; // Visual size scaled
    const latentX = pixelX + pixelSize + 250;
    const latentY = (height - latentSize) / 2;

    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    ctx.fillRect(latentX, latentY, latentSize, latentSize);
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.lineWidth = 2;
    ctx.strokeRect(latentX, latentY, latentSize, latentSize);

    // Draw latent grid
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
    ctx.lineWidth = 1;
    const latentGridSize = latentSize / 4;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(latentX + i * latentGridSize, latentY);
      ctx.lineTo(latentX + i * latentGridSize, latentY + latentSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(latentX, latentY + i * latentGridSize);
      ctx.lineTo(latentX + latentSize, latentY + i * latentGridSize);
      ctx.stroke();
    }

    // Label
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Latent Space', latentX - 5, latentY - 10);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    const latentDim = 1024 / compressionRatio;
    ctx.fillText(`${latentDim} × ${latentDim} × 16`, latentX - 15, latentY + latentSize + 20);
    const compression = (3.1 / ((latentDim * latentDim * 16) / 1000000)).toFixed(0);
    ctx.fillText(`= ${((latentDim * latentDim * 16) / 1000000).toFixed(2)}M (${compression}× smaller)`, latentX - 30, latentY + latentSize + 35);

  }, [compressionRatio]);

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Latent Space: <span className="text-fuchsia-400">Efficient Representation</span>
        </h2>
        <p className="text-gray-400">
          Why SD3 operates in latent space, not pixel space
        </p>
      </div>

      {/* Key Insight */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <div className="flex items-start gap-4">
          <Info className="text-fuchsia-400 mt-1" size={24} />
          <div>
            <h3 className="font-bold text-lg text-fuchsia-300 mb-2">The Latent Trick</h3>
            <p className="text-gray-300">
              Diffusion in pixel space is computationally expensive (1024×1024×3 = 3.1M dimensions!).
              SD3 uses a pretrained <strong className="text-purple-400">VAE</strong> to compress images to a 
              much smaller <strong className="text-blue-400">latent space</strong>, performs diffusion there, 
              then decodes back to pixels. Same quality, ~64× less compute!
            </p>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <canvas
          ref={canvasRef}
          width={600}
          height={300}
          className="w-full rounded-xl mb-4"
        />

        {/* Compression Control */}
        <div className="flex items-center gap-4 px-4">
          <ZoomOut className="text-gray-400" size={20} />
          <input
            type="range"
            min="4"
            max="16"
            step="4"
            value={compressionRatio}
            onChange={(e) => setCompressionRatio(parseInt(e.target.value))}
            className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
          />
          <ZoomIn className="text-gray-400" size={20} />
          <span className="text-sm text-gray-400 w-24">{compressionRatio}× compression</span>
        </div>
      </div>

      {/* VAE Details */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <span className="text-purple-400">↓</span> VAE Encoder
          </h3>
          <div className="space-y-3 text-sm text-gray-300">
            <p><strong>Input:</strong> RGB image [H, W, 3]</p>
            <p><strong>Output:</strong> Latent [H/8, W/8, 16]</p>
            <p><strong>Process:</strong></p>
            <ol className="list-decimal list-inside pl-4 space-y-1 text-gray-400">
              <li>Conv layers downsample spatially</li>
              <li>ResNet blocks extract features</li>
              <li>Attention layers capture global context</li>
              <li>Final conv produces 16-channel latent</li>
            </ol>
          </div>
        </div>

        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <span className="text-rose-400">↑</span> VAE Decoder
          </h3>
          <div className="space-y-3 text-sm text-gray-300">
            <p><strong>Input:</strong> Latent [H/8, W/8, 16]</p>
            <p><strong>Output:</strong> RGB image [H, W, 3]</p>
            <p><strong>Process:</strong></p>
            <ol className="list-decimal list-inside pl-4 space-y-1 text-gray-400">
              <li>Initial conv expands channels</li>
              <li>ResNet blocks process features</li>
              <li>Upsampling layers increase resolution</li>
              <li>Final conv produces RGB output</li>
            </ol>
          </div>
        </div>
      </div>

      {/* SD3 Specific VAE */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">SD3's 16-Channel VAE</h3>
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-fuchsia-400">16</p>
            <p className="text-sm text-gray-400">Latent Channels</p>
            <p className="text-xs text-gray-500">(vs 4 in SD1/2)</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-purple-400">8×</p>
            <p className="text-sm text-gray-400">Spatial Compression</p>
            <p className="text-xs text-gray-500">1024px → 128 tokens</p>
          </div>
          <div className="bg-white/5 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-400">0.13</p>
            <p className="text-sm text-gray-400">Scaling Factor</p>
            <p className="text-xs text-gray-500">Normalizes latents</p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-4">
          <p className="text-sm text-gray-300">
            <strong className="text-purple-300">Why 16 channels?</strong> More channels = more information 
            preserved in latent space. SD3's VAE captures finer details than SD1/2's 4-channel VAE, 
            leading to sharper, more detailed outputs. The trade-off is slightly larger latent tensors.
          </p>
        </div>
      </div>

      {/* Memory Comparison */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-xl font-bold mb-4">Memory & Compute Savings</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="py-3 px-4 text-left text-gray-400">Space</th>
                <th className="py-3 px-4 text-left text-gray-400">Resolution</th>
                <th className="py-3 px-4 text-left text-gray-400">Dimensions</th>
                <th className="py-3 px-4 text-left text-gray-400">Size (FP16)</th>
                <th className="py-3 px-4 text-left text-gray-400">Transformer Tokens</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              <tr>
                <td className="py-3 px-4 text-rose-400">Pixel</td>
                <td className="py-3 px-4">1024×1024</td>
                <td className="py-3 px-4">1024 × 1024 × 3</td>
                <td className="py-3 px-4">6.3 MB</td>
                <td className="py-3 px-4">1,048,576 😱</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-blue-400">Latent</td>
                <td className="py-3 px-4">128×128</td>
                <td className="py-3 px-4">128 × 128 × 16</td>
                <td className="py-3 px-4">0.5 MB</td>
                <td className="py-3 px-4">16,384 ✓</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-green-400">Patched</td>
                <td className="py-3 px-4">64×64 patches</td>
                <td className="py-3 px-4">64 × 64 = 4096</td>
                <td className="py-3 px-4">~32 MB (with dim)</td>
                <td className="py-3 px-4">4,096 🚀</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-4">
          SD3 patches the 128×128 latent into 2×2 patches, yielding 64×64 = 4096 tokens for the transformer.
          This is manageable with modern attention mechanisms.
        </p>
      </div>
    </div>
  );
}
