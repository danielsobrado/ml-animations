import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ZoomIn, Layers } from 'lucide-react';

export default function EncoderLayerPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showResiduals, setShowResiduals] = useState(true);
  const [highlightedComponent, setHighlightedComponent] = useState(null);

  const steps = [
    { title: 'Input Embeddings', description: 'Token, segment, and position embeddings are summed and enter the first encoder layer.' },
    { title: 'Multi-Head Self-Attention', description: '12 attention heads process the input in parallel, each learning different patterns.' },
    { title: 'Add & Normalize (1)', description: 'Residual connection adds original input, then Layer Normalization stabilizes training.' },
    { title: 'Feed-Forward Network', description: 'Two linear layers with GELU activation: 768 → 3072 → 768. Processes each position independently.' },
    { title: 'Add & Normalize (2)', description: 'Another residual connection and Layer Normalization produce the final layer output.' },
    { title: 'Stack 12 Layers', description: 'This entire block repeats 12 times (BERT-Base) or 24 times (BERT-Large).' },
  ];

  const components = [
    { id: 'input', name: 'Input', color: 'blue' },
    { id: 'attention', name: 'Multi-Head Attention', color: 'yellow' },
    { id: 'add1', name: 'Add & Norm', color: 'green' },
    { id: 'ffn', name: 'Feed-Forward', color: 'purple' },
    { id: 'add2', name: 'Add & Norm', color: 'green' },
    { id: 'output', name: 'Output', color: 'orange' },
  ];

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const getOpacity = (componentIndex) => {
    const stepMapping = [0, 1, 2, 3, 4, 5]; // which step activates each component
    return currentStep >= stepMapping[componentIndex] ? 1 : 0.3;
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          BERT <span className="text-yellow-400">Encoder Layer</span> Architecture
        </h2>
        <p className="text-gray-400">
          Inside one Transformer encoder block - repeated 12× in BERT-Base
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying ? 'Pause' : 'Play Animation'}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
        >
          <RotateCcw size={18} />
          Reset
        </button>
        <button
          onClick={() => setShowResiduals(!showResiduals)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${showResiduals ? 'bg-green-600' : 'bg-white/10 hover:bg-white/20'}`}
        >
          <Layers size={18} />
          {showResiduals ? 'Hide' : 'Show'} Residuals
        </button>
      </div>

      {/* Step Progress */}
      <div className="flex flex-wrap justify-center gap-2">
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`px-3 py-1 rounded-full text-sm transition-all ${
              i === currentStep 
                ? 'bg-yellow-500 text-black' 
                : i < currentStep 
                ? 'bg-yellow-900 text-yellow-300' 
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Description */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
        <h3 className="font-bold text-yellow-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Architecture Diagram */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <div className="flex justify-center">
          <div className="relative">
            {/* Stack indicator for step 6 */}
            {currentStep >= 5 && (
              <div className="absolute -left-16 top-1/2 transform -translate-y-1/2">
                <div className="bg-orange-900/30 border border-orange-500 rounded-lg px-2 py-8">
                  <p className="text-xs text-orange-400 writing-mode-vertical transform -rotate-180" style={{ writingMode: 'vertical-rl' }}>
                    × 12 Layers
                  </p>
                </div>
              </div>
            )}

            <div className="flex flex-col items-center gap-2">
              {/* Input */}
              <div 
                className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 0 ? 'bg-blue-900/30 border-blue-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(0) }}
              >
                <p className="text-center text-sm font-medium text-blue-300">Input Embeddings</p>
                <p className="text-center text-xs text-gray-400">[batch, seq_len, 768]</p>
              </div>

              {/* Arrow */}
              <div className="text-gray-500">↓</div>

              {/* Multi-Head Attention */}
              <div 
                className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 1 ? 'bg-yellow-900/30 border-yellow-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(1) }}
              >
                <p className="text-center text-sm font-medium text-yellow-300">Multi-Head Attention</p>
                <p className="text-center text-xs text-gray-400">12 heads × 64 dims</p>
              </div>

              {/* Residual connection 1 */}
              {showResiduals && (
                <div className="absolute left-[-40px] top-[60px] h-[100px] w-[30px]">
                  <svg className="w-full h-full" style={{ opacity: currentStep >= 2 ? 1 : 0.3 }}>
                    <path
                      d="M 30 0 L 0 0 L 0 100 L 30 100"
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="2"
                      strokeDasharray="4"
                    />
                    <polygon points="25,95 30,100 25,105" fill="#22c55e" />
                  </svg>
                </div>
              )}

              {/* Add & Norm 1 */}
              <div className="flex items-center gap-2">
                <div className="text-gray-500">↓</div>
                {showResiduals && <span className="text-green-400 text-xs">+ residual</span>}
              </div>
              <div 
                className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 2 ? 'bg-green-900/30 border-green-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(2) }}
              >
                <p className="text-center text-sm font-medium text-green-300">Add & LayerNorm</p>
                <p className="text-center text-xs text-gray-400">x + Attention(x)</p>
              </div>

              {/* Arrow */}
              <div className="text-gray-500">↓</div>

              {/* Feed-Forward Network */}
              <div 
                className={`w-48 p-4 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 3 ? 'bg-purple-900/30 border-purple-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(3) }}
              >
                <p className="text-center text-sm font-medium text-purple-300">Feed-Forward Network</p>
                <div className="flex flex-col items-center mt-2 text-xs text-gray-400">
                  <span>Linear: 768 → 3072</span>
                  <span className="text-purple-400">GELU activation</span>
                  <span>Linear: 3072 → 768</span>
                </div>
              </div>

              {/* Residual connection 2 */}
              {showResiduals && (
                <div className="absolute left-[-40px] top-[230px] h-[120px] w-[30px]">
                  <svg className="w-full h-full" style={{ opacity: currentStep >= 4 ? 1 : 0.3 }}>
                    <path
                      d="M 30 0 L 0 0 L 0 120 L 30 120"
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth="2"
                      strokeDasharray="4"
                    />
                    <polygon points="25,115 30,120 25,125" fill="#22c55e" />
                  </svg>
                </div>
              )}

              {/* Add & Norm 2 */}
              <div className="flex items-center gap-2">
                <div className="text-gray-500">↓</div>
                {showResiduals && <span className="text-green-400 text-xs">+ residual</span>}
              </div>
              <div 
                className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 4 ? 'bg-green-900/30 border-green-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(4) }}
              >
                <p className="text-center text-sm font-medium text-green-300">Add & LayerNorm</p>
                <p className="text-center text-xs text-gray-400">x + FFN(x)</p>
              </div>

              {/* Arrow */}
              <div className="text-gray-500">↓</div>

              {/* Output */}
              <div 
                className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${
                  currentStep >= 4 ? 'bg-orange-900/30 border-orange-500' : 'bg-gray-800/30 border-gray-600'
                }`}
                style={{ opacity: getOpacity(5) }}
              >
                <p className="text-center text-sm font-medium text-orange-300">Layer Output</p>
                <p className="text-center text-xs text-gray-400">[batch, seq_len, 768]</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Component Details */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Layer Normalization */}
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-400 mb-3">📊 Layer Normalization</h4>
          <p className="text-sm text-gray-300 mb-2">
            Normalizes across the feature dimension (not batch), stabilizing training:
          </p>
          <div className="bg-black/30 rounded-lg p-3">
            <code className="text-xs text-green-300">
              LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
            </code>
          </div>
          <ul className="text-xs text-gray-400 mt-2 space-y-1">
            <li>• μ, σ computed per token (across 768 dims)</li>
            <li>• γ, β are learned parameters</li>
            <li>• Helps with vanishing/exploding gradients</li>
          </ul>
        </div>

        {/* Feed-Forward Network */}
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-400 mb-3">🔧 Feed-Forward Network</h4>
          <p className="text-sm text-gray-300 mb-2">
            Position-wise FFN with expansion factor 4:
          </p>
          <div className="bg-black/30 rounded-lg p-3">
            <code className="text-xs text-purple-300">
              FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
            </code>
          </div>
          <ul className="text-xs text-gray-400 mt-2 space-y-1">
            <li>• W₁: [768, 3072] - expansion</li>
            <li>• W₂: [3072, 768] - contraction</li>
            <li>• GELU: smooth ReLU alternative</li>
            <li>• Processes each position independently</li>
          </ul>
        </div>
      </div>

      {/* Residual Connections */}
      <div className="bg-gradient-to-r from-green-900/20 to-blue-900/20 rounded-xl p-4 border border-green-500/30">
        <h4 className="font-bold text-green-400 mb-3">🔄 Residual Connections (Skip Connections)</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-300 mb-2"><strong>Why Residuals?</strong></p>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• Enable training of very deep networks (12+ layers)</li>
              <li>• Gradient flows directly through skip path</li>
              <li>• Each layer learns a "modification" to input</li>
              <li>• Prevents vanishing gradients</li>
            </ul>
          </div>
          <div>
            <p className="text-sm text-gray-300 mb-2"><strong>Mathematical Form:</strong></p>
            <div className="bg-black/30 rounded-lg p-3">
              <code className="text-sm text-green-300">
                output = LayerNorm(x + SubLayer(x))
              </code>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              BERT uses "post-norm": LayerNorm after residual addition
            </p>
          </div>
        </div>
      </div>

      {/* GELU Activation */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold text-purple-400 mb-3">🌊 GELU Activation (Gaussian Error Linear Unit)</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-300 mb-2">Used instead of ReLU in BERT:</p>
            <div className="bg-black/30 rounded-lg p-3">
              <code className="text-xs text-purple-300">
                GELU(x) = x * Φ(x)<br/>
                ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
              </code>
            </div>
          </div>
          <div>
            <p className="text-sm text-gray-300 mb-2">Properties:</p>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Smooth, non-monotonic</li>
              <li>• Allows small negative values</li>
              <li>• Weights inputs by their magnitude</li>
              <li>• Better for NLP than ReLU</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Full Model Stats */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <h4 className="font-bold text-yellow-400 mb-3">📈 BERT Model Statistics</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/20">
                <th className="text-left py-2 text-gray-400">Component</th>
                <th className="text-center py-2 text-gray-400">BERT-Base</th>
                <th className="text-center py-2 text-gray-400">BERT-Large</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-white/10">
                <td className="py-2">Encoder Layers</td>
                <td className="text-center font-mono">12</td>
                <td className="text-center font-mono">24</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">Hidden Size</td>
                <td className="text-center font-mono">768</td>
                <td className="text-center font-mono">1024</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">Attention Heads</td>
                <td className="text-center font-mono">12</td>
                <td className="text-center font-mono">16</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">FFN Intermediate</td>
                <td className="text-center font-mono">3072</td>
                <td className="text-center font-mono">4096</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2">Total Parameters</td>
                <td className="text-center font-mono text-green-400">110M</td>
                <td className="text-center font-mono text-green-400">340M</td>
              </tr>
              <tr>
                <td className="py-2">Max Sequence Length</td>
                <td className="text-center font-mono">512</td>
                <td className="text-center font-mono">512</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 PyTorch Encoder Layer:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-blue-300">{`class BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config.hidden_size, config.num_heads)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),  # 768 → 3072
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),  # 3072 → 768
            nn.Dropout(config.dropout)
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x, attention_mask=None):
        # Multi-Head Attention + Residual + LayerNorm
        attn_output = self.attention(x, attention_mask)
        x = self.attention_norm(x + attn_output)
        
        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x  # [batch, seq_len, 768]

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            BertEncoderLayer(config) for _ in range(config.num_layers)
        ])
    
    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x`}</code>
        </pre>
      </div>
    </div>
  );
}
