import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Brain, Calculator } from 'lucide-react';

export default function PracticePanel() {
  const [mode, setMode] = useState('quiz');
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answered, setAnswered] = useState(0);

  // Architecture calculator state
  const [inputDim, setInputDim] = useState(784);
  const [hiddenDims, setHiddenDims] = useState([400, 200]);
  const [latentDim, setLatentDim] = useState(20);

  const questions = [
    {
      question: "What does the encoder output in a VAE?",
      options: [
        "A single latent vector z",
        "Mean (μ) and variance (σ²) parameters",
        "The reconstructed image directly",
        "Classification probabilities"
      ],
      correct: 1,
      explanation: "The encoder outputs μ and log σ² (or σ²), which parameterize the latent distribution q(z|x). We then sample z from this distribution."
    },
    {
      question: "What is the reparameterization trick used for?",
      options: [
        "To speed up training",
        "To reduce memory usage",
        "To enable backpropagation through the sampling step",
        "To normalize the inputs"
      ],
      correct: 2,
      explanation: "The reparameterization trick (z = μ + σ⊙ε where ε~N(0,I)) moves the randomness outside the computational graph, allowing gradients to flow through."
    },
    {
      question: "What does the KL divergence term in the VAE loss encourage?",
      options: [
        "Better image quality",
        "Faster training convergence",
        "The latent distribution to be close to N(0, I)",
        "Larger latent dimensions"
      ],
      correct: 2,
      explanation: "KL(q(z|x) || p(z)) measures how different the learned distribution is from the prior N(0,I). Minimizing it regularizes the latent space."
    },
    {
      question: "How can you generate new samples with a trained VAE?",
      options: [
        "Run the encoder on random noise",
        "Sample z from N(0,I) and pass through decoder",
        "Average all training samples",
        "Use the KL divergence directly"
      ],
      correct: 1,
      explanation: "After training, sample z ~ N(0,I) and decode it. The decoder has learned to map points in the standard normal to realistic outputs."
    },
    {
      question: "Why do we output log σ² instead of σ directly?",
      options: [
        "It's faster to compute",
        "Variance must be positive; log can be any real number",
        "It reduces the number of parameters",
        "It improves image quality"
      ],
      correct: 1,
      explanation: "σ² must be positive, but neural network outputs can be any real number. Using log σ² removes this constraint. We recover σ = exp(0.5 × log σ²)."
    },
    {
      question: "What happens if β > 1 in a β-VAE?",
      options: [
        "Training becomes faster",
        "Reconstruction quality improves",
        "Latent representations become more disentangled",
        "The model overfits more"
      ],
      correct: 2,
      explanation: "Higher β puts more weight on the KL term, forcing the latent space to be more Gaussian. This encourages disentanglement but may blur reconstructions."
    },
    {
      question: "What is the 'posterior collapse' problem in VAEs?",
      options: [
        "The decoder produces all zeros",
        "The encoder ignores the input and outputs the prior",
        "The loss becomes negative",
        "Training takes too long"
      ],
      correct: 1,
      explanation: "Posterior collapse occurs when the encoder outputs μ≈0, σ≈1 regardless of input, meaning q(z|x) ≈ p(z). The latent code carries no information about x."
    },
    {
      question: "For MNIST images (28×28), what is a typical latent dimension?",
      options: [
        "784 (same as input)",
        "2 (for visualization only)",
        "10-50",
        "1000+"
      ],
      correct: 2,
      explanation: "For MNIST, 10-50 dimensions work well. 2D is for visualization but limited capacity. Too high dimensions are unnecessary for simple digits."
    },
    {
      question: "What is the ELBO in VAE terminology?",
      options: [
        "A type of activation function",
        "Evidence Lower Bound - the VAE loss function",
        "A regularization technique",
        "A decoder architecture"
      ],
      correct: 1,
      explanation: "ELBO (Evidence Lower Bound) is the objective we maximize. It equals reconstruction log-likelihood minus KL divergence. Maximizing ELBO ≈ minimizing the VAE loss."
    },
    {
      question: "Why are VAE-generated images often blurry compared to GANs?",
      options: [
        "VAEs use smaller networks",
        "VAEs optimize pixel-wise reconstruction, averaging over uncertainty",
        "VAEs train slower",
        "VAEs can only generate grayscale images"
      ],
      correct: 1,
      explanation: "VAEs optimize log-likelihood which averages over all possible outputs. When uncertain, this leads to averaging (blurring). GANs optimize adversarial loss which prefers sharp outputs."
    }
  ];

  const handleAnswer = (index) => {
    if (showResult) return;
    setSelectedAnswer(index);
    setShowResult(true);
    setAnswered(prev => prev + 1);
    if (index === questions[currentQuestion].correct) {
      setScore(prev => prev + 1);
    }
  };

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    }
  };

  const resetQuiz = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setAnswered(0);
  };

  // Calculate parameters
  const calculateParams = () => {
    let total = 0;
    let breakdown = [];
    
    // Encoder
    let prevDim = inputDim;
    hiddenDims.forEach((dim, i) => {
      const params = prevDim * dim + dim; // weights + biases
      breakdown.push({ name: `Encoder FC${i+1}`, params, dims: `${prevDim} → ${dim}` });
      total += params;
      prevDim = dim;
    });
    
    // μ and log σ² layers
    const muParams = prevDim * latentDim + latentDim;
    const logvarParams = prevDim * latentDim + latentDim;
    breakdown.push({ name: 'FC μ', params: muParams, dims: `${prevDim} → ${latentDim}` });
    breakdown.push({ name: 'FC log σ²', params: logvarParams, dims: `${prevDim} → ${latentDim}` });
    total += muParams + logvarParams;
    
    // Decoder (mirror of encoder)
    prevDim = latentDim;
    [...hiddenDims].reverse().forEach((dim, i) => {
      const params = prevDim * dim + dim;
      breakdown.push({ name: `Decoder FC${i+1}`, params, dims: `${prevDim} → ${dim}` });
      total += params;
      prevDim = dim;
    });
    
    // Output layer
    const outParams = prevDim * inputDim + inputDim;
    breakdown.push({ name: 'FC Output', params: outParams, dims: `${prevDim} → ${inputDim}` });
    total += outParams;
    
    return { total, breakdown };
  };

  const { total, breakdown } = calculateParams();

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex justify-center gap-2">
        <button
          onClick={() => setMode('quiz')}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            mode === 'quiz'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Brain size={20} />
          Quiz (10 Questions)
        </button>
        <button
          onClick={() => setMode('calculator')}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            mode === 'calculator'
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Calculator size={20} />
          Parameter Calculator
        </button>
      </div>

      {mode === 'quiz' && (
        <>
          {/* Progress */}
          <div className="flex justify-center gap-1">
            {questions.map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-full transition-all ${
                  i === currentQuestion
                    ? 'bg-purple-500 scale-125'
                    : i < currentQuestion
                    ? 'bg-purple-800'
                    : 'bg-gray-700'
                }`}
              />
            ))}
          </div>

          {/* Score */}
          <div className="text-center">
            <p className="text-gray-400">
              Score: <span className="text-purple-400 font-bold">{score}</span> / {answered}
            </p>
          </div>

          {/* Question Card */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <p className="text-sm text-gray-500 mb-2">
              Question {currentQuestion + 1} of {questions.length}
            </p>
            <h3 className="text-xl font-bold mb-6">
              {questions[currentQuestion].question}
            </h3>

            <div className="space-y-3">
              {questions[currentQuestion].options.map((option, i) => {
                let buttonClass = 'bg-white/5 hover:bg-white/10 border-white/10';
                
                if (showResult) {
                  if (i === questions[currentQuestion].correct) {
                    buttonClass = 'bg-green-900/50 border-green-500';
                  } else if (i === selectedAnswer && i !== questions[currentQuestion].correct) {
                    buttonClass = 'bg-red-900/50 border-red-500';
                  }
                }

                return (
                  <button
                    key={i}
                    onClick={() => handleAnswer(i)}
                    disabled={showResult}
                    className={`w-full text-left p-4 rounded-xl border transition-all ${buttonClass}`}
                  >
                    <div className="flex items-center gap-3">
                      {showResult && i === questions[currentQuestion].correct && (
                        <CheckCircle className="text-green-400" size={20} />
                      )}
                      {showResult && i === selectedAnswer && i !== questions[currentQuestion].correct && (
                        <XCircle className="text-red-400" size={20} />
                      )}
                      <span>{option}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {showResult && (
              <div className="mt-4 p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
                <p className="text-sm text-gray-300">
                  <strong className="text-purple-400">Explanation:</strong> {questions[currentQuestion].explanation}
                </p>
              </div>
            )}

            <div className="flex justify-between mt-6">
              <button
                onClick={resetQuiz}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg"
              >
                <RotateCcw size={18} />
                Reset
              </button>
              {showResult && currentQuestion < questions.length - 1 && (
                <button
                  onClick={nextQuestion}
                  className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg"
                >
                  Next Question →
                </button>
              )}
              {showResult && currentQuestion === questions.length - 1 && (
                <div className="text-purple-400 font-bold">
                  Final Score: {score} / {questions.length}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {mode === 'calculator' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">VAE Parameter Calculator</h3>
            
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div>
                <label className="text-sm text-gray-400">Input Dimension</label>
                <input
                  type="number"
                  value={inputDim}
                  onChange={(e) => setInputDim(parseInt(e.target.value) || 784)}
                  className="w-full mt-1 px-3 py-2 bg-black/30 border border-white/20 rounded-lg"
                />
              </div>
              <div>
                <label className="text-sm text-gray-400">Hidden Dims (comma-sep)</label>
                <input
                  type="text"
                  value={hiddenDims.join(', ')}
                  onChange={(e) => setHiddenDims(e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))}
                  className="w-full mt-1 px-3 py-2 bg-black/30 border border-white/20 rounded-lg"
                />
              </div>
              <div>
                <label className="text-sm text-gray-400">Latent Dimension</label>
                <input
                  type="number"
                  value={latentDim}
                  onChange={(e) => setLatentDim(parseInt(e.target.value) || 20)}
                  className="w-full mt-1 px-3 py-2 bg-black/30 border border-white/20 rounded-lg"
                />
              </div>
            </div>

            {/* Quick Presets */}
            <div className="mb-6">
              <p className="text-sm text-gray-400 mb-2">Quick Presets:</p>
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => { setInputDim(784); setHiddenDims([400, 200]); setLatentDim(20); }}
                  className="px-3 py-1 bg-purple-900/50 hover:bg-purple-800/50 rounded-lg text-sm border border-purple-500/30"
                >
                  MNIST Simple
                </button>
                <button
                  onClick={() => { setInputDim(784); setHiddenDims([512, 256]); setLatentDim(64); }}
                  className="px-3 py-1 bg-purple-900/50 hover:bg-purple-800/50 rounded-lg text-sm border border-purple-500/30"
                >
                  MNIST Deep
                </button>
                <button
                  onClick={() => { setInputDim(3072); setHiddenDims([1024, 512, 256]); setLatentDim(128); }}
                  className="px-3 py-1 bg-purple-900/50 hover:bg-purple-800/50 rounded-lg text-sm border border-purple-500/30"
                >
                  CIFAR-10
                </button>
                <button
                  onClick={() => { setInputDim(12288); setHiddenDims([2048, 1024, 512]); setLatentDim(256); }}
                  className="px-3 py-1 bg-purple-900/50 hover:bg-purple-800/50 rounded-lg text-sm border border-purple-500/30"
                >
                  64×64 RGB
                </button>
              </div>
            </div>

            {/* Parameter Breakdown */}
            <div className="bg-black/20 rounded-xl p-4">
              <h4 className="font-bold mb-3">Parameter Breakdown</h4>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-white/10">
                    <th className="text-left py-2">Layer</th>
                    <th className="text-left py-2">Dimensions</th>
                    <th className="text-right py-2">Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.map((item, i) => (
                    <tr key={i} className="border-b border-white/5">
                      <td className="py-2">{item.name}</td>
                      <td className="py-2 text-gray-400 font-mono text-xs">{item.dims}</td>
                      <td className="py-2 text-right text-purple-400">{item.params.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="font-bold">
                    <td className="py-3" colSpan={2}>Total Parameters</td>
                    <td className="py-3 text-right text-pink-400 text-lg">{total.toLocaleString()}</td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Key Takeaways */}
      <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-2xl p-6 border border-purple-500/30">
        <h3 className="text-lg font-bold mb-4 text-purple-300">🎓 Key Takeaways</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>Encoder outputs μ and σ, not z directly</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>Reparameterization: z = μ + σ⊙ε</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>Loss = Reconstruction + β × KL Divergence</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>Generation: sample z ~ N(0,I), decode</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>Smooth latent space enables interpolation</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-400">✓</span>
            <span>β {'>'} 1 → β-VAE (more disentanglement)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
