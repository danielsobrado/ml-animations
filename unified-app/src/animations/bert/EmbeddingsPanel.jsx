import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Plus, Layers, Info } from 'lucide-react';

export default function EmbeddingsPanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showDimensions, setShowDimensions] = useState(false);

  const tokens = ['[CLS]', 'The', 'cat', 'sat', '[SEP]'];
  
  const steps = [
    { title: 'Token Embeddings', description: 'Each token is mapped to a 768-dimensional vector from a learned vocabulary embedding table.' },
    { title: 'Segment Embeddings', description: 'Indicates which sentence each token belongs to (Sentence A or B). Used for tasks with sentence pairs.' },
    { title: 'Position Embeddings', description: 'Encodes the position of each token in the sequence (0, 1, 2, ...). BERT learns these, unlike sinusoidal encodings.' },
    { title: 'Sum All Three', description: 'All three embeddings are element-wise added together to form the final input embedding.' },
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
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const EmbeddingVector = ({ label, color, values, active, delay = 0 }) => (
    <div 
      className={`transition-all duration-500 ${active ? 'opacity-100 scale-100' : 'opacity-30 scale-95'}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className={`bg-${color}-900/30 border border-${color}-500/30 rounded-lg p-2`}>
        <p className={`text-xs text-${color}-400 font-medium mb-1`}>{label}</p>
        <div className="flex gap-0.5">
          {values.map((v, i) => (
            <div 
              key={i} 
              className={`w-4 h-8 rounded-sm bg-${color}-500`}
              style={{ opacity: 0.3 + v * 0.7 }}
              title={`dim ${i}: ${v.toFixed(2)}`}
            />
          ))}
          <span className="text-gray-500 text-xs ml-1">...</span>
        </div>
      </div>
    </div>
  );

  // Generate pseudo-random but consistent values for visualization
  const genValues = (seed) => {
    const vals = [];
    for (let i = 0; i < 8; i++) {
      vals.push((Math.sin(seed * (i + 1)) + 1) / 2);
    }
    return vals;
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Input Embeddings: <span className="text-yellow-400">Three Components Combined</span>
        </h2>
        <p className="text-gray-400">
          BERT's input is the sum of token, segment, and position embeddings
        </p>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
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
          onClick={() => setShowDimensions(!showDimensions)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${showDimensions ? 'bg-purple-600' : 'bg-white/10 hover:bg-white/20'}`}
        >
          <Layers size={18} />
          {showDimensions ? 'Hide' : 'Show'} Dimensions
        </button>
      </div>

      {/* Step Progress */}
      <div className="flex justify-center gap-2">
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStep(i); setIsPlaying(false); }}
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
              i === currentStep 
                ? 'bg-yellow-500 text-black scale-110' 
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

      {/* Main Visualization */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10 overflow-x-auto">
        <div className="min-w-[700px]">
          {/* Token row */}
          <div className="flex items-center gap-4 mb-6">
            <div className="w-24 text-right text-sm text-gray-400">Tokens:</div>
            <div className="flex gap-4">
              {tokens.map((token, i) => (
                <div key={i} className="w-24 text-center">
                  <span className={`px-3 py-1 rounded font-mono text-sm ${
                    token.startsWith('[') ? 'bg-red-600/30 text-red-300' : 'bg-blue-600/30 text-blue-300'
                  }`}>
                    {token}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Token Embeddings */}
          <div className={`flex items-center gap-4 mb-4 transition-all duration-500 ${currentStep >= 0 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="w-24 text-right text-sm text-blue-400">Token Emb:</div>
            <div className="flex gap-4">
              {tokens.map((token, i) => (
                <div key={i} className="w-24">
                  <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg p-2 h-16 flex flex-col justify-center">
                    <div className="flex gap-0.5 justify-center">
                      {genValues(i + 1).slice(0, 6).map((v, j) => (
                        <div 
                          key={j} 
                          className="w-2 h-6 rounded-sm bg-blue-500"
                          style={{ opacity: 0.3 + v * 0.7 }}
                        />
                      ))}
                    </div>
                    {showDimensions && <p className="text-xs text-center text-gray-500 mt-1">768-d</p>}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">E<sub>token</sub></div>
          </div>

          {/* Plus Sign */}
          {currentStep >= 1 && (
            <div className="flex items-center gap-4 my-2">
              <div className="w-24" />
              <div className="flex gap-4">
                {tokens.map((_, i) => (
                  <div key={i} className="w-24 text-center">
                    <Plus size={16} className="mx-auto text-gray-500" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Segment Embeddings */}
          <div className={`flex items-center gap-4 mb-4 transition-all duration-500 ${currentStep >= 1 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="w-24 text-right text-sm text-green-400">Segment Emb:</div>
            <div className="flex gap-4">
              {tokens.map((token, i) => (
                <div key={i} className="w-24">
                  <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-2 h-16 flex flex-col justify-center">
                    <div className="flex gap-0.5 justify-center">
                      {genValues(100).slice(0, 6).map((v, j) => (
                        <div 
                          key={j} 
                          className="w-2 h-6 rounded-sm bg-green-500"
                          style={{ opacity: 0.3 + v * 0.7 }}
                        />
                      ))}
                    </div>
                    <p className="text-xs text-center text-green-400 mt-1">E<sub>A</sub></p>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">E<sub>segment</sub></div>
          </div>

          {/* Plus Sign */}
          {currentStep >= 2 && (
            <div className="flex items-center gap-4 my-2">
              <div className="w-24" />
              <div className="flex gap-4">
                {tokens.map((_, i) => (
                  <div key={i} className="w-24 text-center">
                    <Plus size={16} className="mx-auto text-gray-500" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Position Embeddings */}
          <div className={`flex items-center gap-4 mb-4 transition-all duration-500 ${currentStep >= 2 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="w-24 text-right text-sm text-purple-400">Position Emb:</div>
            <div className="flex gap-4">
              {tokens.map((token, i) => (
                <div key={i} className="w-24">
                  <div className="bg-purple-900/30 border border-purple-500/30 rounded-lg p-2 h-16 flex flex-col justify-center">
                    <div className="flex gap-0.5 justify-center">
                      {genValues(200 + i * 10).slice(0, 6).map((v, j) => (
                        <div 
                          key={j} 
                          className="w-2 h-6 rounded-sm bg-purple-500"
                          style={{ opacity: 0.3 + v * 0.7 }}
                        />
                      ))}
                    </div>
                    <p className="text-xs text-center text-purple-400 mt-1">pos={i}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">E<sub>position</sub></div>
          </div>

          {/* Equals Sign */}
          {currentStep >= 3 && (
            <div className="flex items-center gap-4 my-4">
              <div className="w-24" />
              <div className="flex gap-4">
                {tokens.map((_, i) => (
                  <div key={i} className="w-24 text-center">
                    <div className="border-t-2 border-white/30 mx-4" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Final Embeddings */}
          <div className={`flex items-center gap-4 transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-30'}`}>
            <div className="w-24 text-right text-sm text-orange-400 font-bold">Input Emb:</div>
            <div className="flex gap-4">
              {tokens.map((token, i) => (
                <div key={i} className="w-24">
                  <div className="bg-orange-900/30 border-2 border-orange-500/50 rounded-lg p-2 h-16 flex flex-col justify-center">
                    <div className="flex gap-0.5 justify-center">
                      {genValues(300 + i * 5).slice(0, 6).map((v, j) => (
                        <div 
                          key={j} 
                          className="w-2 h-6 rounded-sm bg-orange-500"
                          style={{ opacity: 0.5 + v * 0.5 }}
                        />
                      ))}
                    </div>
                    {showDimensions && <p className="text-xs text-center text-gray-500 mt-1">768-d</p>}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-orange-400 font-bold">→ to Encoder</div>
          </div>
        </div>
      </div>

      {/* Formula */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10 text-center">
        <p className="font-mono text-lg">
          <span className="text-orange-400">E<sub>input</sub></span> = 
          <span className="text-blue-400"> E<sub>token</sub></span> + 
          <span className="text-green-400"> E<sub>segment</sub></span> + 
          <span className="text-purple-400"> E<sub>position</sub></span>
        </p>
        <p className="text-sm text-gray-500 mt-2">All embeddings are 768-dimensional vectors (for BERT-Base)</p>
      </div>

      {/* Detailed Explanations */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
          <h4 className="font-bold text-blue-400 mb-2">🔤 Token Embeddings</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Lookup table: vocab_size × 768</li>
            <li>• ~30,522 tokens in BERT vocab</li>
            <li>• Each token has unique embedding</li>
            <li>• Learned during pre-training</li>
          </ul>
        </div>
        <div className="bg-green-900/20 rounded-xl p-4 border border-green-500/30">
          <h4 className="font-bold text-green-400 mb-2">📑 Segment Embeddings</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Only 2 vectors: E<sub>A</sub> and E<sub>B</sub></li>
            <li>• E<sub>A</sub> for first sentence</li>
            <li>• E<sub>B</sub> for second sentence</li>
            <li>• Used in NSP, QA, NLI tasks</li>
          </ul>
        </div>
        <div className="bg-purple-900/20 rounded-xl p-4 border border-purple-500/30">
          <h4 className="font-bold text-purple-400 mb-2">📍 Position Embeddings</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• Learned (not sinusoidal)</li>
            <li>• Max 512 positions</li>
            <li>• Encodes word order</li>
            <li>• Position 0 is [CLS]</li>
          </ul>
        </div>
      </div>

      {/* Two-Sentence Example */}
      <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Info size={20} className="text-blue-400" />
          Sentence Pair Example (for NSP, QA, etc.)
        </h3>
        <div className="overflow-x-auto">
          <div className="flex gap-2 min-w-max items-end">
            {['[CLS]', 'The', 'cat', 'sat', '[SEP]', 'It', 'was', 'happy', '[SEP]'].map((token, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <span className={`px-2 py-1 rounded text-xs font-mono ${
                  token.startsWith('[') ? 'bg-red-600/30 text-red-300' : 
                  i <= 4 ? 'bg-blue-600/30 text-blue-300' : 'bg-green-600/30 text-green-300'
                }`}>
                  {token}
                </span>
                <span className={`text-xs ${i <= 4 ? 'text-blue-400' : 'text-green-400'}`}>
                  {i <= 4 ? 'Seg A' : 'Seg B'}
                </span>
                <span className="text-xs text-purple-400">pos {i}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 PyTorch Implementation:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-yellow-300">{`class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token embeddings: vocab_size × hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # Position embeddings: max_position × hidden_size
        self.position_embeddings = nn.Embedding(config.max_position, config.hidden_size)
        # Segment embeddings: 2 × hidden_size (Sentence A or B)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids, token_type_ids, position_ids):
        # Get all three embeddings
        word_embeds = self.word_embeddings(input_ids)        # [B, L, 768]
        position_embeds = self.position_embeddings(position_ids)  # [B, L, 768]
        segment_embeds = self.token_type_embeddings(token_type_ids)  # [B, L, 768]
        
        # Sum them up
        embeddings = word_embeds + position_embeds + segment_embeds
        
        # LayerNorm + Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # [B, L, 768]`}</code>
        </pre>
      </div>
    </div>
  );
}
