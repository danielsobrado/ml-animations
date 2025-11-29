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
      question: "What is the key difference between joint attention and cross-attention?",
      options: [
        "Joint attention is faster",
        "In joint attention, all tokens can attend to all tokens bidirectionally",
        "Cross-attention has more parameters",
        "Joint attention only works with images"
      ],
      correct: 1,
      explanation: "Joint attention concatenates image and text tokens into one sequence, allowing bidirectional information flow. Cross-attention only allows image to query text."
    },
    {
      question: "How does text token behavior differ in joint vs cross attention?",
      options: [
        "Text tokens are discarded in both",
        "Text tokens update in joint attention, stay frozen in cross-attention",
        "Text tokens are always frozen",
        "Text tokens update in cross-attention only"
      ],
      correct: 1,
      explanation: "In joint attention, text tokens participate fully and get updated each layer. In cross-attention, text tokens are read-only (frozen)."
    },
    {
      question: "What is the attention complexity for a joint sequence of N_img + N_txt tokens?",
      options: [
        "O(N_img × N_txt)",
        "O(N_img + N_txt)",
        "O((N_img + N_txt)²)",
        "O(N_img²)"
      ],
      correct: 2,
      explanation: "Self-attention on the joint sequence has O(N²) complexity where N = N_img + N_txt, meaning the attention matrix is (N_img + N_txt)²."
    },
    {
      question: "For a 1024×1024 image with 2×2 patching, how many image tokens are there?",
      options: [
        "1024",
        "4096",
        "16384",
        "262144"
      ],
      correct: 1,
      explanation: "1024px / 8 (VAE) = 128 latent pixels. With 2×2 patches: (128/2)² = 64² = 4096 image tokens."
    },
    {
      question: "What technique reduces attention memory from O(N²) to O(N)?",
      options: [
        "Cross-attention",
        "Token pruning",
        "Flash Attention",
        "Quantization"
      ],
      correct: 2,
      explanation: "Flash Attention uses tiling and recomputation to avoid materializing the full N×N attention matrix, reducing memory to O(N)."
    },
    {
      question: "How many types of attention interactions exist in joint attention?",
      options: [
        "1 (self-attention only)",
        "2 (image-image, text-text)",
        "3 (self, cross, image)",
        "4 (img→img, txt→txt, img→txt, txt→img)"
      ],
      correct: 3,
      explanation: "Joint attention creates four interaction types: image self-attention, text self-attention, and bidirectional cross-modal attention (image↔text)."
    },
    {
      question: "Why do text tokens benefit from seeing the image state?",
      options: [
        "To reduce memory usage",
        "To adapt text meaning based on what's actually forming",
        "To speed up inference",
        "It doesn't help - it's just simpler to implement"
      ],
      correct: 1,
      explanation: "When text tokens see the image, they can adapt their representation. 'Cat' might emphasize different features if a realistic vs cartoon cat is forming."
    },
    {
      question: "In SD3's MM-DiT, what happens to image and text after joint attention?",
      options: [
        "They're kept together forever",
        "They're split back into separate streams",
        "Text tokens are discarded",
        "Only image tokens continue"
      ],
      correct: 1,
      explanation: "After joint attention, the sequence is split back into image and text portions. Each goes through separate MLPs before the next layer."
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
          Practice Lab: <span className="text-violet-400">Joint Attention</span>
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
              ? 'bg-violet-600 text-white'
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
              ? 'bg-violet-600 text-white'
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
              {/* Progress */}
              <div className="flex justify-between items-center mb-6">
                <span className="text-gray-400">
                  Question {currentQuestion + 1} of {questions.length}
                </span>
                <span className="text-violet-400">Score: {score}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2 mb-8">
                <div
                  className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-2 rounded-full transition-all"
                  style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                />
              </div>

              {/* Question */}
              <h3 className="text-xl font-bold mb-6">
                {questions[currentQuestion].question}
              </h3>

              {/* Options */}
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

              {/* Explanation */}
              {showResult && (
                <div className="mt-6 p-4 rounded-xl bg-violet-500/20 border border-violet-500/30">
                  <p className="text-sm">
                    <span className="font-bold text-violet-300">Explanation:</span>{' '}
                    {questions[currentQuestion].explanation}
                  </p>
                </div>
              )}

              {/* Next Button */}
              {showResult && (
                <button
                  onClick={nextQuestion}
                  className="mt-6 px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-xl transition-colors"
                >
                  {currentQuestion < questions.length - 1 ? 'Next Question' : 'See Results'}
                </button>
              )}
            </>
          ) : (
            /* Results */
            <div className="text-center py-8">
              <Trophy className="w-20 h-20 mx-auto mb-6 text-yellow-400" />
              <h3 className="text-2xl font-bold mb-4">Quiz Complete!</h3>
              <p className="text-4xl font-bold text-violet-400 mb-4">
                {score} / {questions.length}
              </p>
              <p className="text-gray-400 mb-6">
                {score === questions.length
                  ? "Perfect! You've mastered joint attention!"
                  : score >= questions.length * 0.7
                  ? "Great job! You understand the core concepts."
                  : "Keep learning! Review the panels and try again."}
              </p>
              <button
                onClick={resetQuiz}
                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-xl transition-colors flex items-center gap-2 mx-auto"
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
              <span className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center text-sm">1</span>
              Implement Basic Joint Attention
            </h3>
            <p className="text-gray-400 mb-4">
              Create a simple joint attention mechanism from scratch.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class JointAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # TODO: Define Q, K, V projections
        # TODO: Define output projection
        
    def forward(self, img_tokens, txt_tokens):
        # TODO: Concatenate tokens
        # joint = torch.cat([img_tokens, txt_tokens], dim=1)
        
        # TODO: Project to Q, K, V
        
        # TODO: Compute attention (scaled dot-product)
        
        # TODO: Split back into img and txt portions
        
        pass

# Test your implementation
batch_size = 2
img_seq = 64  # 8x8 patches
txt_seq = 10
d_model = 256

img = torch.randn(batch_size, img_seq, d_model)
txt = torch.randn(batch_size, txt_seq, d_model)

# attn = JointAttention(d_model, num_heads=8)
# img_out, txt_out = attn(img, txt)
# print(f"img_out: {img_out.shape}, txt_out: {txt_out.shape}")`}</pre>
            </div>
            <div className="mt-4 p-4 bg-green-500/10 rounded-xl border border-green-500/30">
              <p className="text-sm text-green-300">
                <strong>Goal:</strong> Understand how joint attention is just self-attention on concatenated sequences.
              </p>
            </div>
          </div>

          {/* Exercise 2 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center text-sm">2</span>
              Visualize Attention Patterns
            </h3>
            <p className="text-gray-400 mb-4">
              Extract and visualize the four attention pattern types.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import matplotlib.pyplot as plt
import seaborn as sns

def visualize_joint_attention(attn_weights, n_img, n_txt):
    """
    attn_weights: (n_img + n_txt, n_img + n_txt) attention matrix
    n_img: number of image tokens
    n_txt: number of text tokens
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO: Extract four quadrants
    # img_to_img = attn_weights[:n_img, :n_img]
    # img_to_txt = attn_weights[:n_img, n_img:]
    # txt_to_img = attn_weights[n_img:, :n_img]
    # txt_to_txt = attn_weights[n_img:, n_img:]
    
    # TODO: Plot each quadrant as a heatmap
    # Use sns.heatmap() or plt.imshow()
    
    # TODO: Add titles and labels
    # "Image→Image", "Image→Text", "Text→Image", "Text→Text"
    
    plt.tight_layout()
    plt.savefig("joint_attention_patterns.png")

# Create synthetic attention weights for testing
n_img, n_txt = 64, 10
total = n_img + n_txt
fake_attn = torch.softmax(torch.randn(total, total), dim=-1)
# visualize_joint_attention(fake_attn.numpy(), n_img, n_txt)`}</pre>
            </div>
            <div className="mt-4 p-4 bg-purple-500/10 rounded-xl border border-purple-500/30">
              <p className="text-sm text-purple-300">
                <strong>Goal:</strong> See the four different attention patterns in a joint attention matrix.
              </p>
            </div>
          </div>

          {/* Exercise 3 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm">3</span>
              Compare Memory Usage
            </h3>
            <p className="text-gray-400 mb-4">
              Benchmark memory usage of joint vs cross attention.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
import gc

def measure_attention_memory(attn_type, n_img, n_txt, d_model):
    """Measure peak memory for attention computation."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    img = torch.randn(1, n_img, d_model, device='cuda')
    txt = torch.randn(1, n_txt, d_model, device='cuda')
    
    if attn_type == 'joint':
        # TODO: Implement joint attention
        # joint = torch.cat([img, txt], dim=1)
        # attn = F.scaled_dot_product_attention(joint, joint, joint)
        pass
    else:
        # TODO: Implement cross attention
        # cross = F.scaled_dot_product_attention(img, txt, txt)
        # self_attn = F.scaled_dot_product_attention(img, img, img)
        pass
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
    return peak_mem

# Test different sequence lengths
results = []
for n_img in [256, 1024, 4096]:
    n_txt = 77
    d_model = 1536
    
    # joint_mem = measure_attention_memory('joint', n_img, n_txt, d_model)
    # cross_mem = measure_attention_memory('cross', n_img, n_txt, d_model)
    # results.append((n_img, joint_mem, cross_mem))
    
# TODO: Plot results showing memory scaling`}</pre>
            </div>
            <div className="mt-4 p-4 bg-blue-500/10 rounded-xl border border-blue-500/30">
              <p className="text-sm text-blue-300">
                <strong>Goal:</strong> Understand the quadratic memory scaling of joint attention.
              </p>
            </div>
          </div>

          {/* Exercise 4 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-rose-600 flex items-center justify-center text-sm">4</span>
              Extract Real Attention from SD3
            </h3>
            <p className="text-gray-400 mb-4">
              Hook into SD3's MM-DiT to visualize real attention patterns.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention
import torch

# Storage for attention maps
attention_maps = {}

def create_attention_hook(name):
    def hook(module, input, output):
        # The attention module computes attention internally
        # We need to capture the attention weights before softmax
        # This requires modifying the attention processor
        pass
    return hook

def register_hooks(pipe):
    """Register hooks on all attention layers."""
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            module.register_forward_hook(create_attention_hook(name))

# pipe = StableDiffusion3Pipeline.from_pretrained(...)
# register_hooks(pipe)

# Run inference
# image = pipe("a cat on the beach", num_inference_steps=1)

# TODO: Analyze attention_maps
# - Which text tokens get highest attention from image regions?
# - How does attention change across layers?
# - Do certain image regions specialize for certain words?`}</pre>
            </div>
            <div className="mt-4 p-4 bg-rose-500/10 rounded-xl border border-rose-500/30">
              <p className="text-sm text-rose-300">
                <strong>Goal:</strong> See how SD3 actually uses joint attention during generation.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Key Takeaways */}
      <div className="bg-gradient-to-r from-violet-900/30 to-fuchsia-900/30 rounded-2xl p-6 border border-violet-500/30">
        <h3 className="text-xl font-bold mb-4">Key Takeaways</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-violet-400 mt-2" />
            <p className="text-gray-300">
              <strong>Joint attention</strong> concatenates image and text tokens, then applies self-attention
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-fuchsia-400 mt-2" />
            <p className="text-gray-300">
              <strong>Bidirectional flow</strong> allows text to see and adapt to the forming image
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-blue-400 mt-2" />
            <p className="text-gray-300">
              <strong>Quadratic complexity</strong> O((N_img + N_txt)²) is the cost of full interaction
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-green-400 mt-2" />
            <p className="text-gray-300">
              <strong>Flash Attention</strong> makes joint attention practical at scale
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
