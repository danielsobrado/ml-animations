import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Eye, EyeOff, Target, BookOpen } from 'lucide-react';

export default function PreTrainingPanel() {
  const [currentTask, setCurrentTask] = useState('mlm');
  const [mlmStep, setMlmStep] = useState(0);
  const [nspStep, setNspStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showMaskedTokens, setShowMaskedTokens] = useState(false);

  const mlmSteps = [
    { title: 'Original Sentence', description: 'Start with a sentence from the training corpus.' },
    { title: 'Random Masking (15%)', description: '15% of tokens are selected for masking. Of these: 80% → [MASK], 10% → random word, 10% → unchanged.' },
    { title: 'Predict Masked Tokens', description: 'BERT predicts the original tokens at masked positions using bidirectional context.' },
    { title: 'Compute Loss', description: 'Cross-entropy loss only for masked tokens. Learn to understand language structure.' },
  ];

  const nspSteps = [
    { title: 'Sample Sentence Pairs', description: 'Create pairs: 50% consecutive sentences (IsNext), 50% random sentences (NotNext).' },
    { title: 'Add Special Tokens', description: '[CLS] + Sentence A + [SEP] + Sentence B + [SEP]. Segment embeddings distinguish A and B.' },
    { title: 'Predict Relationship', description: 'Use [CLS] token output to predict if B follows A (binary classification).' },
    { title: 'Learn Document Understanding', description: 'Model learns inter-sentence relationships useful for QA, NLI tasks.' },
  ];

  const mlmExample = {
    original: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    masked: ['The', '[MASK]', 'brown', '[MASK]', 'jumps', 'over', 'the', 'lazy', '[MASK]'],
    predictions: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    maskedIndices: [1, 3, 8]
  };

  const nspExample = {
    positive: {
      sentA: 'The cat sat on the mat.',
      sentB: 'It was very comfortable.',
      label: 'IsNext',
      isPositive: true
    },
    negative: {
      sentA: 'The cat sat on the mat.',
      sentB: 'Python is a programming language.',
      label: 'NotNext',
      isPositive: false
    }
  };

  useEffect(() => {
    if (isPlaying) {
      const steps = currentTask === 'mlm' ? mlmSteps : nspSteps;
      const setStep = currentTask === 'mlm' ? setMlmStep : setNspStep;
      const currentStepVal = currentTask === 'mlm' ? mlmStep : nspStep;

      const interval = setInterval(() => {
        if (currentStepVal >= steps.length - 1) {
          setIsPlaying(false);
        } else {
          setStep(prev => prev + 1);
        }
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isPlaying, currentTask, mlmStep, nspStep]);

  const reset = () => {
    setMlmStep(0);
    setNspStep(0);
    setIsPlaying(false);
  };

  const currentSteps = currentTask === 'mlm' ? mlmSteps : nspSteps;
  const currentStepIndex = currentTask === 'mlm' ? mlmStep : nspStep;
  const setCurrentStepIndex = currentTask === 'mlm' ? setMlmStep : setNspStep;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          BERT <span className="text-yellow-400">Pre-Training</span> Objectives
        </h2>
        <p className="text-gray-400">
          Two self-supervised tasks: Masked Language Model (MLM) & Next Sentence Prediction (NSP)
        </p>
      </div>

      {/* Task Selection */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => { setCurrentTask('mlm'); reset(); }}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all ${
            currentTask === 'mlm' 
              ? 'bg-yellow-600 text-white scale-105' 
              : 'bg-white/10 hover:bg-white/20'
          }`}
        >
          <EyeOff size={20} />
          <div className="text-left">
            <p className="font-bold">MLM</p>
            <p className="text-xs opacity-75">Masked Language Model</p>
          </div>
        </button>
        <button
          onClick={() => { setCurrentTask('nsp'); reset(); }}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all ${
            currentTask === 'nsp' 
              ? 'bg-purple-600 text-white scale-105' 
              : 'bg-white/10 hover:bg-white/20'
          }`}
        >
          <BookOpen size={20} />
          <div className="text-left">
            <p className="font-bold">NSP</p>
            <p className="text-xs opacity-75">Next Sentence Prediction</p>
          </div>
        </button>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
            currentTask === 'mlm' ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-purple-600 hover:bg-purple-700'
          }`}
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
      </div>

      {/* Step Progress */}
      <div className="flex justify-center gap-2">
        {currentSteps.map((_, i) => (
          <button
            key={i}
            onClick={() => { setCurrentStepIndex(i); setIsPlaying(false); }}
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
              i === currentStepIndex 
                ? currentTask === 'mlm' ? 'bg-yellow-500 text-black scale-110' : 'bg-purple-500 text-white scale-110'
                : i < currentStepIndex 
                ? currentTask === 'mlm' ? 'bg-yellow-900 text-yellow-300' : 'bg-purple-900 text-purple-300'
                : 'bg-white/10 text-gray-500'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      {/* Current Step Description */}
      <div className={`rounded-xl p-4 border ${
        currentTask === 'mlm' 
          ? 'bg-yellow-900/20 border-yellow-500/30' 
          : 'bg-purple-900/20 border-purple-500/30'
      }`}>
        <h3 className={`font-bold ${currentTask === 'mlm' ? 'text-yellow-400' : 'text-purple-400'}`}>
          Step {currentStepIndex + 1}: {currentSteps[currentStepIndex].title}
        </h3>
        <p className="text-gray-300 mt-1">{currentSteps[currentStepIndex].description}</p>
      </div>

      {/* MLM Visualization */}
      {currentTask === 'mlm' && (
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-lg font-bold mb-4 text-yellow-400 flex items-center gap-2">
            <EyeOff size={20} />
            Masked Language Model (MLM)
          </h3>

          {/* Token Visualization */}
          <div className="space-y-6">
            {/* Original */}
            <div className={`transition-all duration-500 ${mlmStep >= 0 ? 'opacity-100' : 'opacity-30'}`}>
              <p className="text-sm text-gray-400 mb-2">Original sentence:</p>
              <div className="flex flex-wrap gap-2">
                {mlmExample.original.map((token, i) => (
                  <span 
                    key={i} 
                    className="px-3 py-2 bg-blue-600/30 rounded text-blue-300 font-mono"
                  >
                    {token}
                  </span>
                ))}
              </div>
            </div>

            {/* Masked */}
            <div className={`transition-all duration-500 ${mlmStep >= 1 ? 'opacity-100' : 'opacity-30'}`}>
              <p className="text-sm text-gray-400 mb-2">After masking (15% tokens):</p>
              <div className="flex flex-wrap gap-2">
                {mlmExample.masked.map((token, i) => {
                  const isMasked = mlmExample.maskedIndices.includes(i);
                  return (
                    <span 
                      key={i} 
                      className={`px-3 py-2 rounded font-mono ${
                        isMasked 
                          ? 'bg-red-600/50 text-red-300 border-2 border-red-500' 
                          : 'bg-gray-700/30 text-gray-400'
                      }`}
                    >
                      {token}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* BERT Prediction */}
            <div className={`transition-all duration-500 ${mlmStep >= 2 ? 'opacity-100' : 'opacity-30'}`}>
              <p className="text-sm text-gray-400 mb-2">BERT predicts:</p>
              <div className="flex flex-wrap gap-2">
                {mlmExample.predictions.map((token, i) => {
                  const isMasked = mlmExample.maskedIndices.includes(i);
                  return (
                    <div key={i} className="flex flex-col items-center">
                      <span 
                        className={`px-3 py-2 rounded font-mono ${
                          isMasked 
                            ? 'bg-green-600/50 text-green-300 border-2 border-green-500' 
                            : 'bg-gray-700/30 text-gray-400'
                        }`}
                      >
                        {token}
                      </span>
                      {isMasked && mlmStep >= 3 && (
                        <span className="text-xs text-green-400 mt-1">✓ correct</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Loss */}
            {mlmStep >= 3 && (
              <div className="bg-orange-900/20 border border-orange-500/30 rounded-lg p-3">
                <p className="text-sm text-orange-300">
                  <strong>MLM Loss:</strong> Cross-entropy computed only for masked positions
                </p>
                <code className="text-xs text-gray-400 mt-1 block">
                  Loss = -Σ log P(original_token | context) for masked positions only
                </code>
              </div>
            )}
          </div>

          {/* Masking Strategy */}
          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="bg-red-900/20 rounded-lg p-3 border border-red-500/30">
              <p className="text-2xl font-bold text-red-400">80%</p>
              <p className="text-xs text-gray-400">Replace with [MASK]</p>
            </div>
            <div className="bg-orange-900/20 rounded-lg p-3 border border-orange-500/30">
              <p className="text-2xl font-bold text-orange-400">10%</p>
              <p className="text-xs text-gray-400">Replace with random</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-500/30">
              <p className="text-2xl font-bold text-gray-400">10%</p>
              <p className="text-xs text-gray-400">Keep unchanged</p>
            </div>
          </div>
        </div>
      )}

      {/* NSP Visualization */}
      {currentTask === 'nsp' && (
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-lg font-bold mb-4 text-purple-400 flex items-center gap-2">
            <BookOpen size={20} />
            Next Sentence Prediction (NSP)
          </h3>

          {/* Examples */}
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            {/* Positive Example */}
            <div className={`p-4 rounded-xl border-2 transition-all duration-500 ${
              nspStep >= 0 ? 'bg-green-900/20 border-green-500/50' : 'bg-gray-800/30 border-gray-600/30'
            }`}>
              <p className="text-sm font-medium text-green-400 mb-2">✓ Positive (50%): IsNext</p>
              <div className={`transition-all ${nspStep >= 1 ? 'opacity-100' : 'opacity-50'}`}>
                <p className="text-xs text-gray-400">[CLS]</p>
                <p className="text-blue-300 text-sm">{nspExample.positive.sentA}</p>
                <p className="text-xs text-gray-400">[SEP]</p>
                <p className="text-green-300 text-sm">{nspExample.positive.sentB}</p>
                <p className="text-xs text-gray-400">[SEP]</p>
              </div>
              {nspStep >= 2 && (
                <div className="mt-3 bg-green-600/30 rounded px-2 py-1 inline-block">
                  <span className="text-sm text-green-300">Prediction: IsNext ✓</span>
                </div>
              )}
            </div>

            {/* Negative Example */}
            <div className={`p-4 rounded-xl border-2 transition-all duration-500 ${
              nspStep >= 0 ? 'bg-red-900/20 border-red-500/50' : 'bg-gray-800/30 border-gray-600/30'
            }`}>
              <p className="text-sm font-medium text-red-400 mb-2">✗ Negative (50%): NotNext</p>
              <div className={`transition-all ${nspStep >= 1 ? 'opacity-100' : 'opacity-50'}`}>
                <p className="text-xs text-gray-400">[CLS]</p>
                <p className="text-blue-300 text-sm">{nspExample.negative.sentA}</p>
                <p className="text-xs text-gray-400">[SEP]</p>
                <p className="text-orange-300 text-sm">{nspExample.negative.sentB}</p>
                <p className="text-xs text-gray-400">[SEP]</p>
              </div>
              {nspStep >= 2 && (
                <div className="mt-3 bg-red-600/30 rounded px-2 py-1 inline-block">
                  <span className="text-sm text-red-300">Prediction: NotNext ✓</span>
                </div>
              )}
            </div>
          </div>

          {/* CLS Token Usage */}
          {nspStep >= 2 && (
            <div className="bg-purple-900/20 border border-purple-500/30 rounded-lg p-4">
              <p className="text-sm text-purple-300 font-medium mb-2">How NSP works:</p>
              <ol className="text-xs text-gray-400 space-y-1 list-decimal list-inside">
                <li>The [CLS] token aggregates information from both sentences</li>
                <li>[CLS] output → Linear layer → Binary classification (IsNext/NotNext)</li>
                <li>Binary cross-entropy loss for this 2-class prediction</li>
              </ol>
            </div>
          )}
        </div>
      )}

      {/* Pre-Training Details */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-yellow-900/20 rounded-xl p-4 border border-yellow-500/30">
          <h4 className="font-bold text-yellow-400 mb-3">📚 Pre-Training Data</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• <strong>BookCorpus:</strong> ~800M words (11,038 books)</li>
            <li>• <strong>English Wikipedia:</strong> ~2,500M words</li>
            <li>• Document-level corpus for NSP task</li>
            <li>• Total: ~3.3 billion words</li>
          </ul>
        </div>
        <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30">
          <h4 className="font-bold text-blue-400 mb-3">⚙️ Pre-Training Config</h4>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>• <strong>Batch size:</strong> 256 sequences</li>
            <li>• <strong>Max length:</strong> 512 tokens</li>
            <li>• <strong>Steps:</strong> 1,000,000</li>
            <li>• <strong>Time:</strong> ~4 days on 16 TPU chips</li>
            <li>• <strong>Optimizer:</strong> Adam (lr=1e-4)</li>
          </ul>
        </div>
      </div>

      {/* Combined Loss */}
      <div className="bg-gradient-to-r from-yellow-900/20 to-purple-900/20 rounded-xl p-4 border border-white/20">
        <h4 className="font-bold text-white mb-3">🎯 Total Pre-Training Loss</h4>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <code className="text-lg">
            <span className="text-white">L</span>
            <span className="text-gray-400">_total</span>
            <span className="text-white"> = </span>
            <span className="text-yellow-400">L</span>
            <span className="text-yellow-300">_MLM</span>
            <span className="text-white"> + </span>
            <span className="text-purple-400">L</span>
            <span className="text-purple-300">_NSP</span>
          </code>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          Both losses are weighted equally. MLM provides language understanding, NSP provides discourse understanding.
        </p>
      </div>

      {/* Note about NSP */}
      <div className="bg-orange-900/20 rounded-xl p-4 border border-orange-500/30">
        <h4 className="font-bold text-orange-400 mb-2">⚠️ Note on NSP</h4>
        <p className="text-sm text-gray-300">
          Later research (RoBERTa, ALBERT) found that NSP may not be necessary and can even hurt performance.
          These models use only MLM or modified versions like Sentence Order Prediction (SOP).
        </p>
      </div>

      {/* Code Example */}
      <div className="bg-black/40 rounded-xl p-4 border border-white/10">
        <p className="text-sm text-gray-400 mb-3">🐍 PyTorch Pre-Training Heads:</p>
        <pre className="text-sm overflow-x-auto">
          <code className="text-cyan-300">{`class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLM Head: predict masked tokens
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size)  # 768 → 30522
        )
        # NSP Head: binary classification from [CLS]
        self.nsp_head = nn.Linear(config.hidden_size, 2)  # 768 → 2
    
    def forward(self, sequence_output, pooled_output):
        # sequence_output: [batch, seq_len, 768] - for MLM
        # pooled_output: [batch, 768] - [CLS] token output for NSP
        
        mlm_logits = self.mlm_head(sequence_output)  # [batch, seq_len, vocab_size]
        nsp_logits = self.nsp_head(pooled_output)    # [batch, 2]
        
        return mlm_logits, nsp_logits

# Training loop (simplified)
def compute_loss(model, batch):
    outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
    mlm_logits, nsp_logits = outputs
    
    # MLM loss (only for masked positions)
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, vocab_size),
        batch['mlm_labels'].view(-1),
        ignore_index=-100  # Ignore non-masked tokens
    )
    
    # NSP loss
    nsp_loss = F.cross_entropy(nsp_logits, batch['nsp_labels'])
    
    return mlm_loss + nsp_loss`}</code>
        </pre>
      </div>
    </div>
  );
}
