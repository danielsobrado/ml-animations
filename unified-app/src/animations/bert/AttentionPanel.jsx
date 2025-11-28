import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Eye, Layers } from 'lucide-react';

export default function AttentionPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedHead, setSelectedHead] = useState(0);
  const [showAllHeads, setShowAllHeads] = useState(false);

  const tokens = ['[CLS]', 'The', 'cat', 'sat', 'on', 'mat'];
  
  const steps = [
    { title: 'Input Embeddings', description: 'Each token embedding (768-d) enters the self-attention layer.' },
    { title: 'Linear Projections', description: 'Project embeddings to Q, K, V matrices. For each head: 768 → 64 dimensions.' },
    { title: 'Compute Attention Scores', description: 'Score = Q × Kᵀ / √dₖ. Each token queries all other tokens to determine relevance.' },
    { title: 'Apply Softmax', description: 'Softmax normalizes scores to attention weights (sum to 1 for each query token).' },
    { title: 'Weighted Sum of Values', description: 'Output = attention_weights × V. Aggregate information based on relevance.' },
    { title: 'Concatenate Heads', description: 'Concat all 12 heads (12 × 64 = 768), then project with Wᴼ back to 768-d.' },
  ];

  // Attention patterns for different heads (simplified visualization)
  const headPatterns = [
    { name: 'Position', description: 'Attends to nearby tokens', pattern: (i, j) => 1 - Math.abs(i - j) * 0.2 },
    { name: '[CLS] Focus', description: 'All tokens attend to [CLS]', pattern: (i, j) => j === 0 ? 0.8 : 0.1 },
    { name: 'Syntax', description: 'Subject-verb relationships', pattern: (i, j) => (i === 1 && j === 3) || (i === 3 && j === 1) ? 0.9 : 0.15 },
    { name: 'Self', description: 'Token attends to itself', pattern: (i, j) => i === j ? 0.7 : 0.15 },
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

  const AttentionMatrix = ({ pattern, title, size = 'large' }) => {
    const cellSize = size === 'large' ? 'w-10 h-10' : 'w-6 h-6';
    const textSize = size === 'large' ? 'text-xs' : 'text-[8px]';
    
    return (
      <div className="inline-block">
        {title && <p className="text-xs text-gray-400 mb-1 text-center">{title}</p>}
        <div className="flex flex-col">
          <div className="flex">
            <div className={`${cellSize}`} />
            {tokens.map((t, i) => (
              <div key={i} className={`${cellSize} ${textSize} flex items-center justify-center text-gray-500 font-mono`}>
                {t.slice(0, 3)}
              </div>
            ))}
          </div>
          {tokens.map((rowToken, i) => (
            <div key={i} className="flex">
              <div className={`${cellSize} ${textSize} flex items-center justify-center text-gray-500 font-mono`}>
                {rowToken.slice(0, 3)}
              </div>
              {tokens.map((_, j) => {
                const value = pattern(i, j);
                return (
                  <div
                    key={j}
                    className={`${cellSize} flex items-center justify-center ${textSize} rounded-sm m-0.5`}
                    style={{ 
                      backgroundColor: `rgba(234, 179, 8, ${value})`,
                      color: value > 0.5 ? 'black' : 'white'
                    }}
                  >
                    {size === 'large' ? value.toFixed(2) : ''}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Multi-Head Self-Attention in <span className="text-yellow-400">BERT</span>
        </h2>
        <p className="text-gray-400">
          12 attention heads (BERT-Base) learn different types of relationships
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
          onClick={() => setShowAllHeads(!showAllHeads)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${showAllHeads ? 'bg-purple-600' : 'bg-white/10 hover:bg-white/20'}`}
        >
          <Layers size={18} />
          {showAllHeads ? 'Single Head' : 'All Heads'}
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
            {i + 1}. {step.title}
          </button>
        ))}
      </div>

      {/* Current Step Description */}
      <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4">
        <h3 className="font-bold text-yellow-400">Step {currentStep + 1}: {steps[currentStep].title}</h3>
        <p className="text-gray-300 mt-1">{steps[currentStep].description}</p>
      </div>

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10 overflow-x-auto">
        {/* Flow Diagram */}
        <div className="flex items-center justify-center gap-4 mb-8 min-w-max">
          {/* Input */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 0 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="bg-blue-600/30 border border-blue-500 rounded-lg p-3">
              <p className="text-sm font-medium text-blue-300">Input</p>
              <p className="text-xs text-gray-400">[6, 768]</p>
            </div>
          </div>
          <span className="text-gray-500">→</span>
          
          {/* Q, K, V Projections */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 1 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="flex gap-2">
              <div className="bg-red-600/30 border border-red-500 rounded-lg p-2">
                <p className="text-xs text-red-300">Q</p>
              </div>
              <div className="bg-green-600/30 border border-green-500 rounded-lg p-2">
                <p className="text-xs text-green-300">K</p>
              </div>
              <div className="bg-purple-600/30 border border-purple-500 rounded-lg p-2">
                <p className="text-xs text-purple-300">V</p>
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-1">[6, 12, 64]</p>
          </div>
          <span className="text-gray-500">→</span>
          
          {/* Attention Scores */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 2 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="bg-orange-600/30 border border-orange-500 rounded-lg p-3">
              <p className="text-sm font-medium text-orange-300">Q×Kᵀ/√64</p>
              <p className="text-xs text-gray-400">[6, 6]</p>
            </div>
          </div>
          <span className="text-gray-500">→</span>
          
          {/* Softmax */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="bg-yellow-600/30 border border-yellow-500 rounded-lg p-3">
              <p className="text-sm font-medium text-yellow-300">Softmax</p>
              <p className="text-xs text-gray-400">weights</p>
            </div>
          </div>
          <span className="text-gray-500">→</span>
          
          {/* Weighted Sum */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 4 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="bg-teal-600/30 border border-teal-500 rounded-lg p-3">
              <p className="text-sm font-medium text-teal-300">Attn × V</p>
              <p className="text-xs text-gray-400">[6, 64]</p>
            </div>
          </div>
          <span className="text-gray-500">→</span>
          
          {/* Concat & Project */}
          <div className={`text-center transition-all duration-500 ${currentStep >= 5 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="bg-pink-600/30 border border-pink-500 rounded-lg p-3">
              <p className="text-sm font-medium text-pink-300">Concat + Wᴼ</p>
              <p className="text-xs text-gray-400">[6, 768]</p>
            </div>
          </div>
        </div>

        {/* Attention Matrix Visualization */}
        {currentStep >= 2 && (
          <div className="mt-6">
            {showAllHeads ? (
              <div>
                <h4 className="text-center text-gray-400 mb-4">All 12 Attention Heads (showing 4 representative patterns)</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 justify-items-center">
                  {headPatterns.map((head, i) => (
                    <div key={i} className="text-center">
                      <p className="text-sm text-yellow-400 mb-1">Head {i + 1}: {head.name}</p>
                      <AttentionMatrix pattern={head.pattern} size="small" />
                      <p className="text-xs text-gray-500 mt-1">{head.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <div className="flex gap-2 mb-4">
                  {headPatterns.map((head, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedHead(i)}
                      className={`px-3 py-1 rounded text-sm transition-all ${
                        selectedHead === i ? 'bg-yellow-500 text-black' : 'bg-white/10 hover:bg-white/20'
                      }`}
                    >
                      {head.name}
                    </button>
                  ))}
                </div>
                <AttentionMatrix 
                  pattern={headPatterns[selectedHead].pattern} 
                  title={`Attention Pattern: ${headPatterns[selectedHead].description}`}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Multi-Head Explanation */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
          <h4 className="font-bold text-yellow-400 mb-3">🧠 Why Multiple Heads?</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>Different heads learn different patterns (syntax, semantics, position)</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>Parallel computation - all heads process simultaneously</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>Low-rank approximation: 12 × 64 = 768, but more expressive</span>
            </li>
            <li className="flex gap-2">
              <span className="text-green-400">✓</span>
              <span>Redundancy provides robustness</span>
            </li>
          </ul>
        </div>
        
        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
          <h4 className="font-bold text-purple-400 mb-3">📊 BERT Attention Stats</h4>
          <table className="w-full text-sm">
            <tbody>
              <tr className="border-b border-white/10">
                <td className="py-2 text-gray-400">Number of heads</td>
                <td className="py-2 text-right font-mono">12 (Base) / 16 (Large)</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2 text-gray-400">Head dimension (dₖ)</td>
                <td className="py-2 text-right font-mono">64</td>
              </tr>
              <tr className="border-b border-white/10">
                <td className="py-2 text-gray-400">Total dimension</td>
                <td className="py-2 text-right font-mono">768 (Base) / 1024 (Large)</td>
              </tr>
              <tr>
                <td className="py-2 text-gray-400">Attention layers</td>
                <td className="py-2 text-right font-mono">12 (Base) / 24 (Large)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Bidirectional Attention */}
      <div className="bg-gradient-to-r from-blue-900/30 to-green-900/30 rounded-xl p-4 border border-blue-500/30">
        <h4 className="font-bold text-blue-400 mb-3">🔄 Bidirectional Attention (BERT's Key Innovation)</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium text-gray-300 mb-2">GPT (Unidirectional):</p>
            <div className="flex gap-1">
              {['The', 'cat', 'sat', 'on'].map((t, i) => (
                <div key={i} className="relative">
                  <span className="px-2 py-1 bg-gray-700 rounded text-xs">{t}</span>
                  <div className="text-xs text-gray-500 mt-1">
                    sees: {['The', 'cat', 'sat', 'on'].slice(0, i + 1).join(', ')}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">Each token only sees tokens to its left</p>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-300 mb-2">BERT (Bidirectional):</p>
            <div className="flex gap-1">
              {['The', 'cat', 'sat', 'on'].map((t, i) => (
                <div key={i} className="relative">
                  <span className="px-2 py-1 bg-yellow-700 rounded text-xs">{t}</span>
                  <div className="text-xs text-yellow-500 mt-1">
                    sees: all tokens
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-yellow-500 mt-2">Each token sees ALL other tokens (full context)</p>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 PyTorch Multi-Head Attention:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-green-300">{`class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 64
        
        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # Output projection
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, attention_mask=None):
        B, L, _ = x.shape  # batch, length, hidden_size
        
        # Project to Q, K, V
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, num_heads, L, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask (for padding)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out(context)  # [B, L, 768]`}</code>
        </pre>
      </div>
    </div>
  );
}
