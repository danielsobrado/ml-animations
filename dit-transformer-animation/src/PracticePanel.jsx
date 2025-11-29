import React, { useState } from 'react';
import { CheckCircle, XCircle, Trophy, BookOpen, Code, RotateCcw } from 'lucide-react';

export default function PracticePanel() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [quizComplete, setQuizComplete] = useState(false);
  const [showExercise, setShowExercise] = useState(false);

  const questions = [
    {
      question: "What does DiT stand for?",
      options: [
        "Distributed Image Transformer",
        "Diffusion Transformer",
        "Deep Image Training",
        "Denoising Isotropic Transformer"
      ],
      correct: 1,
      explanation: "DiT stands for Diffusion Transformer - a transformer-based architecture for diffusion models that replaces the U-Net."
    },
    {
      question: "What is the main innovation of AdaLN (Adaptive Layer Normalization)?",
      options: [
        "It's faster than regular LayerNorm",
        "It uses learned γ, β conditioned on timestep/class",
        "It doesn't require gradients",
        "It works without normalization"
      ],
      correct: 1,
      explanation: "AdaLN learns to produce scale (γ) and shift (β) parameters as functions of conditioning (timestep, class, text), allowing dynamic modulation."
    },
    {
      question: "Why is zero-initialization important in AdaLN-Zero?",
      options: [
        "To make training faster",
        "To reduce memory usage",
        "To start as identity and gradually learn each layer's contribution",
        "To prevent NaN gradients"
      ],
      correct: 2,
      explanation: "Zero-initializing the gate parameters makes each block start as identity, allowing residual connections to flow and enabling stable training of deep networks."
    },
    {
      question: "How does DiT inject text conditioning compared to U-Net?",
      options: [
        "Both use only cross-attention",
        "DiT uses cross-attention, U-Net uses AdaLN",
        "DiT uses AdaLN + joint attention, U-Net uses cross-attention",
        "Neither uses text conditioning"
      ],
      correct: 2,
      explanation: "SD3's DiT uses AdaLN for global context (pooled CLIP) and joint attention for sequence-level text. U-Net uses cross-attention to query text tokens."
    },
    {
      question: "What is the 'patchify' step in DiT?",
      options: [
        "Adding noise to patches",
        "Dividing the latent into non-overlapping patches as tokens",
        "Creating image patches for data augmentation",
        "Interpolating between patches"
      ],
      correct: 1,
      explanation: "Patchify divides the latent representation into non-overlapping patches (e.g., 2×2), which become the token sequence for the transformer."
    },
    {
      question: "According to DiT's scaling laws, what happens when you double compute?",
      options: [
        "FID score roughly halves (improves)",
        "Training time doubles with no quality gain",
        "The model becomes unstable",
        "Memory usage decreases"
      ],
      correct: 0,
      explanation: "DiT follows predictable scaling laws where doubling compute approximately halves the FID error, similar to language model scaling."
    },
    {
      question: "What is the patch size typically used in SD3?",
      options: [
        "1×1",
        "2×2",
        "4×4",
        "16×16"
      ],
      correct: 1,
      explanation: "SD3 uses 2×2 patch size on the latent space, meaning 4 latent values (2×2×16 channels) become one token."
    },
    {
      question: "Why did DiT replace U-Net as the backbone?",
      options: [
        "U-Net is patented",
        "DiT is smaller and faster",
        "DiT scales better with more parameters",
        "U-Net doesn't work with diffusion"
      ],
      correct: 2,
      explanation: "Transformers (DiT) scale predictably with more parameters - larger models consistently improve quality, unlike U-Nets which hit diminishing returns."
    }
  ];

  const handleAnswer = (answerIndex) => {
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    if (answerIndex === questions[currentQuestion].correct) {
      setScore(score + 1);
    }
  };

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      setQuizComplete(true);
    }
  };

  const resetQuiz = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setQuizComplete(false);
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Practice Lab: <span className="text-pink-400">DiT Architecture</span>
        </h2>
        <p className="text-gray-400">
          Test your understanding and explore hands-on exercises
        </p>
      </div>

      {/* Tab Toggle */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setShowExercise(false)}
          className={`px-6 py-3 rounded-xl flex items-center gap-2 transition-all ${
            !showExercise
              ? 'bg-pink-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <BookOpen size={18} />
          Quiz
        </button>
        <button
          onClick={() => setShowExercise(true)}
          className={`px-6 py-3 rounded-xl flex items-center gap-2 transition-all ${
            showExercise
              ? 'bg-pink-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Code size={18} />
          Code Exercises
        </button>
      </div>

      {!showExercise ? (
        /* Quiz Section */
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          {!quizComplete ? (
            <>
              <div className="flex justify-between items-center mb-6">
                <span className="text-gray-400">
                  Question {currentQuestion + 1} of {questions.length}
                </span>
                <span className="text-pink-400">Score: {score}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2 mb-8">
                <div
                  className="bg-gradient-to-r from-pink-500 to-orange-500 h-2 rounded-full transition-all"
                  style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                />
              </div>

              <h3 className="text-xl font-bold mb-6">
                {questions[currentQuestion].question}
              </h3>

              <div className="space-y-3">
                {questions[currentQuestion].options.map((option, index) => (
                  <button
                    key={index}
                    onClick={() => !showResult && handleAnswer(index)}
                    disabled={showResult}
                    className={`w-full p-4 rounded-xl text-left transition-all ${
                      showResult
                        ? index === questions[currentQuestion].correct
                          ? 'bg-green-500/30 border-green-500'
                          : index === selectedAnswer
                          ? 'bg-red-500/30 border-red-500'
                          : 'bg-white/5 border-white/10'
                        : 'bg-white/10 border-white/20 hover:bg-white/20'
                    } border`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center font-bold">
                        {String.fromCharCode(65 + index)}
                      </span>
                      <span>{option}</span>
                      {showResult && index === questions[currentQuestion].correct && (
                        <CheckCircle className="ml-auto text-green-400" size={20} />
                      )}
                      {showResult && index === selectedAnswer && index !== questions[currentQuestion].correct && (
                        <XCircle className="ml-auto text-red-400" size={20} />
                      )}
                    </div>
                  </button>
                ))}
              </div>

              {showResult && (
                <div className="mt-6 p-4 rounded-xl bg-pink-500/20 border border-pink-500/30">
                  <p className="text-sm">
                    <span className="font-bold text-pink-300">Explanation:</span>{' '}
                    {questions[currentQuestion].explanation}
                  </p>
                </div>
              )}

              {showResult && (
                <button
                  onClick={nextQuestion}
                  className="mt-6 px-6 py-3 bg-pink-600 hover:bg-pink-500 rounded-xl transition-colors"
                >
                  {currentQuestion < questions.length - 1 ? 'Next Question' : 'See Results'}
                </button>
              )}
            </>
          ) : (
            <div className="text-center py-8">
              <Trophy className="w-20 h-20 mx-auto mb-6 text-yellow-400" />
              <h3 className="text-2xl font-bold mb-4">Quiz Complete!</h3>
              <p className="text-4xl font-bold text-pink-400 mb-4">
                {score} / {questions.length}
              </p>
              <p className="text-gray-400 mb-6">
                {score === questions.length
                  ? "Perfect! You've mastered DiT architecture!"
                  : score >= questions.length * 0.7
                  ? "Great job! You understand the core concepts."
                  : "Keep learning! Review the panels and try again."}
              </p>
              <button
                onClick={resetQuiz}
                className="px-6 py-3 bg-pink-600 hover:bg-pink-500 rounded-xl transition-colors flex items-center gap-2 mx-auto"
              >
                <RotateCcw size={18} />
                Retry Quiz
              </button>
            </div>
          )}
        </div>
      ) : (
        /* Exercises Section */
        <div className="space-y-6">
          {/* Exercise 1 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-pink-600 flex items-center justify-center text-sm">1</span>
              Implement AdaLN from Scratch
            </h3>
            <p className="text-gray-400 mb-4">
              Build an Adaptive Layer Normalization module.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
import torch.nn as nn

class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        # TODO: Create a LayerNorm without learnable affine
        # self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # TODO: Create MLP to produce gamma, beta, alpha
        # Output size should be 3 * hidden_size
        # self.modulation = nn.Sequential(...)
        
        # TODO: Zero-initialize the output layer
        pass
    
    def forward(self, x, c):
        # x: (B, N, D) - features
        # c: (B, D) - conditioning
        
        # TODO: Get gamma, beta, alpha from modulation MLP
        # TODO: Apply adaptive normalization
        # return normalized_x, alpha
        pass

# Test
B, N, D = 2, 100, 256
x = torch.randn(B, N, D)
c = torch.randn(B, D)
# adaln = AdaLayerNorm(D, D)
# out, alpha = adaln(x, c)
# print(f"Output shape: {out.shape}, Alpha shape: {alpha.shape}")`}</pre>
            </div>
          </div>

          {/* Exercise 2 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-orange-600 flex items-center justify-center text-sm">2</span>
              Implement Patchify/Unpatchify
            </h3>
            <p className="text-gray-400 mb-4">
              Convert between spatial latent and sequence of patch tokens.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
import torch.nn as nn

def patchify(x, patch_size=2):
    """
    Convert (B, C, H, W) latent to (B, N, patch_dim) tokens
    where N = (H/patch_size) * (W/patch_size)
    and patch_dim = C * patch_size * patch_size
    """
    B, C, H, W = x.shape
    # TODO: Reshape to extract patches
    # Hint: Use reshape and permute
    pass

def unpatchify(tokens, H, W, patch_size=2):
    """
    Convert (B, N, patch_dim) tokens back to (B, C, H, W) latent
    """
    B, N, patch_dim = tokens.shape
    # TODO: Reshape back to spatial
    pass

# Test
B, C, H, W = 2, 16, 64, 64  # SD3-like latent
patch_size = 2
x = torch.randn(B, C, H, W)

# tokens = patchify(x, patch_size)
# print(f"Tokens shape: {tokens.shape}")  # Should be (2, 1024, 64)

# reconstructed = unpatchify(tokens, H, W, patch_size)
# print(f"Reconstructed shape: {reconstructed.shape}")  # Should be (2, 16, 64, 64)
# print(f"Matches original: {torch.allclose(x, reconstructed)}")`}</pre>
            </div>
          </div>

          {/* Exercise 3 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center text-sm">3</span>
              Build a Simple DiT Block
            </h3>
            <p className="text-gray-400 mb-4">
              Combine attention, MLP, and AdaLN into a full DiT block.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # TODO: Create two AdaLN layers
        # self.norm1 = AdaLayerNorm(hidden_size)
        # self.norm2 = AdaLayerNorm(hidden_size)
        
        # TODO: Create multi-head self-attention
        # self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # TODO: Create MLP (hidden_size -> mlp_ratio*hidden_size -> hidden_size)
        # self.mlp = nn.Sequential(...)
        
        pass
    
    def forward(self, x, c):
        # x: (B, N, D) - features
        # c: (B, D) - conditioning
        
        # TODO: Pre-norm attention with residual and gating
        # norm_x, alpha1 = self.norm1(x, c)
        # attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        # x = x + alpha1.unsqueeze(1) * attn_out
        
        # TODO: Pre-norm MLP with residual and gating
        # norm_x, alpha2 = self.norm2(x, c)
        # mlp_out = self.mlp(norm_x)
        # x = x + alpha2.unsqueeze(1) * mlp_out
        
        return x

# Test
# block = DiTBlock(256, num_heads=8)
# x = torch.randn(2, 100, 256)
# c = torch.randn(2, 256)
# out = block(x, c)
# print(f"Output shape: {out.shape}")`}</pre>
            </div>
          </div>

          {/* Exercise 4 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-cyan-600 flex items-center justify-center text-sm">4</span>
              Compute FLOPs for DiT Variants
            </h3>
            <p className="text-gray-400 mb-4">
              Calculate the computational cost for different model sizes.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`def compute_dit_flops(
    seq_len: int,      # Number of tokens
    hidden_size: int,  # Model dimension
    num_heads: int,    # Attention heads
    num_layers: int,   # Number of blocks
    mlp_ratio: float = 4.0
) -> float:
    """
    Compute approximate FLOPs for a DiT forward pass.
    """
    # Attention FLOPs per layer:
    # - Q, K, V projections: 3 * seq_len * hidden_size^2
    # - Attention scores: 2 * seq_len^2 * hidden_size
    # - Output projection: seq_len * hidden_size^2
    
    # TODO: Calculate attention FLOPs
    attn_flops = 0
    
    # MLP FLOPs per layer:
    # - Up projection: seq_len * hidden_size * (mlp_ratio * hidden_size)
    # - Down projection: same
    
    # TODO: Calculate MLP FLOPs
    mlp_flops = 0
    
    # Total per layer
    layer_flops = attn_flops + mlp_flops
    
    # Total for all layers
    total_flops = num_layers * layer_flops
    
    return total_flops / 1e9  # Return in GFLOPs

# Compare DiT variants
models = {
    'DiT-S/2': {'hidden': 384, 'heads': 6, 'layers': 12},
    'DiT-B/2': {'hidden': 768, 'heads': 12, 'layers': 12},
    'DiT-L/2': {'hidden': 1024, 'heads': 16, 'layers': 24},
    'DiT-XL/2': {'hidden': 1152, 'heads': 16, 'layers': 28},
}

seq_len = 256  # 16x16 patches from 256x256 image with patch_size=16
for name, config in models.items():
    # gflops = compute_dit_flops(seq_len, config['hidden'], 
    #                           config['heads'], config['layers'])
    # print(f"{name}: {gflops:.2f} GFLOPs")
    pass`}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Key Takeaways */}
      <div className="bg-gradient-to-r from-pink-900/30 to-orange-900/30 rounded-2xl p-6 border border-pink-500/30">
        <h3 className="text-xl font-bold mb-4">Key Takeaways</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-pink-400 mt-2" />
            <p className="text-gray-300">
              <strong>DiT</strong> replaces U-Net with a simple, uniform transformer architecture
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-orange-400 mt-2" />
            <p className="text-gray-300">
              <strong>AdaLN-Zero</strong> injects conditioning by modulating layer norm parameters
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-violet-400 mt-2" />
            <p className="text-gray-300">
              <strong>Patchify</strong> converts images to sequences, enabling transformer processing
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-green-400 mt-2" />
            <p className="text-gray-300">
              DiT follows <strong>scaling laws</strong> - larger models consistently improve quality
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
