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
      question: "What are the three text encoders used in SD3?",
      options: [
        "BERT, GPT-2, T5",
        "CLIP-L, CLIP-G, T5-XXL",
        "RoBERTa, ALBERT, CLIP",
        "GPT-4, CLIP, BERT"
      ],
      correct: 1,
      explanation: "SD3 uses CLIP ViT-L/14, CLIP ViT-bigG/14, and T5-XXL (optionally) for comprehensive text understanding."
    },
    {
      question: "What is the main architectural innovation in SD3's denoising backbone?",
      options: [
        "Cross-attention only UNet",
        "Self-attention transformer",
        "MM-DiT with joint attention",
        "Pure convolutional network"
      ],
      correct: 2,
      explanation: "SD3 introduces Multimodal Diffusion Transformer (MM-DiT) that uses joint attention between image and text tokens."
    },
    {
      question: "What compression ratio does the SD3 VAE achieve?",
      options: [
        "4x (256→64 pixels)",
        "8x (512→64 pixels)",
        "16x (1024→64 pixels)",
        "2x (512→256 pixels)"
      ],
      correct: 1,
      explanation: "The VAE compresses images by 8x spatially (e.g., 1024×1024 → 128×128 latents with 16 channels)."
    },
    {
      question: "What sampling method does SD3 primarily use?",
      options: [
        "DDPM",
        "DDIM",
        "Flow matching with Euler",
        "DPM++ 2M"
      ],
      correct: 2,
      explanation: "SD3 uses flow matching (rectified flow) with a simple Euler solver for efficient, high-quality sampling."
    },
    {
      question: "How does SD3 handle guidance (CFG)?",
      options: [
        "Single conditional prediction",
        "Dual predictions (cond + uncond) combined",
        "Triple predictions (cond + partial + uncond)",
        "No guidance needed"
      ],
      correct: 2,
      explanation: "SD3 can use triple classifier-free guidance with conditional, partially-conditional (text only), and unconditional predictions."
    },
    {
      question: "What is the latent dimension (channels) in SD3?",
      options: [
        "4 channels",
        "8 channels",
        "16 channels",
        "32 channels"
      ],
      correct: 2,
      explanation: "SD3 uses 16-channel latents (compared to 4 in SD1.5/SDXL), allowing for more information encoding."
    },
    {
      question: "Why does SD3 use timestep shifting?",
      options: [
        "To reduce memory usage",
        "To handle different resolutions dynamically",
        "To speed up inference only",
        "To reduce model size"
      ],
      correct: 1,
      explanation: "Timestep shifting adjusts the noise schedule based on resolution - higher res needs more noise at early steps."
    },
    {
      question: "What does the pooled text embedding provide?",
      options: [
        "Token-level details",
        "Global context for timestep conditioning",
        "Image feature matching",
        "Noise prediction scaling"
      ],
      correct: 1,
      explanation: "The pooled embedding (from CLIP) provides global semantic context, added to timestep embedding for conditioning."
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
          Practice Lab: <span className="text-fuchsia-400">SD3 Architecture</span>
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
              ? 'bg-fuchsia-600 text-white'
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
              ? 'bg-fuchsia-600 text-white'
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
                <span className="text-fuchsia-400">Score: {score}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2 mb-8">
                <div
                  className="bg-gradient-to-r from-fuchsia-500 to-purple-500 h-2 rounded-full transition-all"
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
                <div className="mt-6 p-4 rounded-xl bg-fuchsia-500/20 border border-fuchsia-500/30">
                  <p className="text-sm">
                    <span className="font-bold text-fuchsia-300">Explanation:</span>{' '}
                    {questions[currentQuestion].explanation}
                  </p>
                </div>
              )}

              {/* Next Button */}
              {showResult && (
                <button
                  onClick={nextQuestion}
                  className="mt-6 px-6 py-3 bg-fuchsia-600 hover:bg-fuchsia-500 rounded-xl transition-colors"
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
              <p className="text-4xl font-bold text-fuchsia-400 mb-4">
                {score} / {questions.length}
              </p>
              <p className="text-gray-400 mb-6">
                {score === questions.length
                  ? "Perfect! You've mastered SD3 architecture!"
                  : score >= questions.length * 0.7
                  ? "Great job! You have a solid understanding."
                  : "Keep learning! Review the other panels and try again."}
              </p>
              <button
                onClick={resetQuiz}
                className="px-6 py-3 bg-fuchsia-600 hover:bg-fuchsia-500 rounded-xl transition-colors flex items-center gap-2 mx-auto"
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
              <span className="w-8 h-8 rounded-lg bg-fuchsia-600 flex items-center justify-center text-sm">1</span>
              Explore the VAE Latent Space
            </h3>
            <p className="text-gray-400 mb-4">
              Encode an image to latents, visualize the 16 channels, and decode back.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`from diffusers import AutoencoderKL
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Load VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae"
).to("cuda")

# Load and encode image
image = Image.open("your_image.png").resize((1024, 1024))
# TODO: Convert to tensor and normalize to [-1, 1]
# TODO: Encode with vae.encode()
# TODO: Get latent representation
# TODO: Visualize first 4 channels
# TODO: Decode back and compare`}</pre>
            </div>
            <div className="mt-4 p-4 bg-green-500/10 rounded-xl border border-green-500/30">
              <p className="text-sm text-green-300">
                <strong>Goal:</strong> Understand how the 16-channel latent space captures image information.
              </p>
            </div>
          </div>

          {/* Exercise 2 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center text-sm">2</span>
              Compare Text Encodings
            </h3>
            <p className="text-gray-400 mb-4">
              Compare how CLIP and T5 encode the same prompt differently.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`from transformers import CLIPTextModel, T5EncoderModel
from transformers import CLIPTokenizer, T5Tokenizer

# Load models
clip_model = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14"
)
t5_model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")

prompt = "A serene mountain lake at sunset"
# TODO: Tokenize with both tokenizers
# TODO: Encode with both models
# TODO: Compare embedding shapes
# TODO: Analyze token-level attention patterns
# TODO: Test with long vs short prompts`}</pre>
            </div>
            <div className="mt-4 p-4 bg-purple-500/10 rounded-xl border border-purple-500/30">
              <p className="text-sm text-purple-300">
                <strong>Goal:</strong> Understand why SD3 uses multiple text encoders and their complementary roles.
              </p>
            </div>
          </div>

          {/* Exercise 3 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm">3</span>
              Implement Timestep Shifting
            </h3>
            <p className="text-gray-400 mb-4">
              Implement resolution-dependent timestep shifting for flow matching.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import numpy as np
import matplotlib.pyplot as plt

def compute_shift(resolution: int, base_res: int = 256) -> float:
    """Compute timestep shift based on resolution."""
    # TODO: Calculate shift factor
    # Higher resolutions need more noise early
    pass

def shift_timesteps(t: np.ndarray, shift: float) -> np.ndarray:
    """Apply shift to timesteps."""
    # TODO: Implement shifted sampling
    # t_shifted = shift * t / (1 + (shift - 1) * t)
    pass

# Test with different resolutions
resolutions = [512, 768, 1024, 1536]
timesteps = np.linspace(0, 1, 1000)

# TODO: Plot original vs shifted timesteps
# TODO: Analyze how shift affects denoising schedule`}</pre>
            </div>
            <div className="mt-4 p-4 bg-blue-500/10 rounded-xl border border-blue-500/30">
              <p className="text-sm text-blue-300">
                <strong>Goal:</strong> Understand how timestep shifting enables resolution-independent training.
              </p>
            </div>
          </div>

          {/* Exercise 4 */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 rounded-lg bg-rose-600 flex items-center justify-center text-sm">4</span>
              Visualize Joint Attention
            </h3>
            <p className="text-gray-400 mb-4">
              Extract and visualize attention maps from the MM-DiT blocks.
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-300">{`import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers"
).to("cuda")

# Hook to capture attention maps
attention_maps = []

def attention_hook(module, input, output):
    # TODO: Extract attention weights
    # TODO: Store for visualization
    pass

# TODO: Register hooks on transformer blocks
# TODO: Run inference
# TODO: Visualize text→image and image→text attention
# TODO: Identify which words attend to which regions`}</pre>
            </div>
            <div className="mt-4 p-4 bg-rose-500/10 rounded-xl border border-rose-500/30">
              <p className="text-sm text-rose-300">
                <strong>Goal:</strong> See how joint attention creates bidirectional text-image understanding.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Key Takeaways */}
      <div className="bg-gradient-to-r from-fuchsia-900/30 to-purple-900/30 rounded-2xl p-6 border border-fuchsia-500/30">
        <h3 className="text-xl font-bold mb-4">Key Takeaways</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-fuchsia-400 mt-2" />
            <p className="text-gray-300">
              SD3 uses <strong>3 text encoders</strong> (CLIP-L, CLIP-G, T5-XXL) for rich prompt understanding
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-purple-400 mt-2" />
            <p className="text-gray-300">
              <strong>MM-DiT</strong> replaces UNet with transformers using joint attention
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-blue-400 mt-2" />
            <p className="text-gray-300">
              <strong>Flow matching</strong> enables efficient training and straight sampling paths
            </p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-rose-400 mt-2" />
            <p className="text-gray-300">
              <strong>16-channel VAE</strong> provides better image reconstruction quality
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
