import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Brain, Calculator, Code } from 'lucide-react';

export default function PracticePanel() {
  const [mode, setMode] = useState('quiz');
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answered, setAnswered] = useState(0);

  // Interactive calculator state
  const [calcT, setCalcT] = useState(0.5);
  const [calcMean, setCalcMean] = useState(0);
  const [calcStd, setCalcStd] = useState(1);

  const questions = [
    {
      question: "In flow matching, what does the neural network predict?",
      options: [
        "The noise ε to be removed",
        "The velocity v_θ(x_t, t) at each point",
        "The final clean image directly",
        "The score function ∇log p(x)"
      ],
      correct: 1,
      explanation: "Flow matching networks predict the velocity field v_θ(x_t, t) that defines how to move through probability space from noise to data."
    },
    {
      question: "What is the main advantage of flow matching over DDPM?",
      options: [
        "It uses less memory",
        "It always produces sharper images",
        "It has a simpler training objective and enables flexible noise schedules",
        "It requires no neural network"
      ],
      correct: 2,
      explanation: "Flow matching has a direct MSE loss on velocity prediction, and the straight-path formulation allows for flexible noise schedules."
    },
    {
      question: "In the Euler method for sampling, what happens at each step?",
      options: [
        "We subtract predicted noise from the image",
        "We add the velocity times step size: x += v * dt",
        "We multiply by the noise schedule",
        "We sample new noise"
      ],
      correct: 1,
      explanation: "The Euler method follows the ODE: x_{t+dt} = x_t + v_θ(x_t, t) × dt, adding the velocity times the timestep."
    },
    {
      question: "Why does logit-normal sampling focus on middle timesteps?",
      options: [
        "Because the network is faster there",
        "Because edge timesteps have less learnable signal",
        "To save computation time",
        "It's just a convention, no real reason"
      ],
      correct: 1,
      explanation: "Near t=0, images are almost pure noise with little structure. Near t=1, they're almost clean. The most challenging denoising happens in the middle."
    },
    {
      question: "What is the formula for linear interpolation in conditional flow matching?",
      options: [
        "x_t = x₀ × x₁",
        "x_t = (1-t)x₀ + tx₁",
        "x_t = x₀ - t × x₁",
        "x_t = exp(-t) × x₀"
      ],
      correct: 1,
      explanation: "Linear interpolation: x_t = (1-t)x₀ + tx₁, where x₀ is noise and x₁ is the target image. At t=0 we have noise, at t=1 we have data."
    },
    {
      question: "What is the target velocity for training when using linear interpolation?",
      options: [
        "v* = x₀ (the noise)",
        "v* = x₁ (the data)",
        "v* = x₁ - x₀ (data minus noise)",
        "v* = (x₁ + x₀) / 2"
      ],
      correct: 2,
      explanation: "For linear paths, the velocity is constant: v* = d(x_t)/dt = x₁ - x₀. The network learns to predict this direction at each point."
    },
    {
      question: "What does the Heun method do differently from Euler?",
      options: [
        "Uses smaller step sizes",
        "Predicts velocity at both start and end of step, then averages",
        "Only works at certain timesteps",
        "Doesn't use a neural network"
      ],
      correct: 1,
      explanation: "Heun (a predictor-corrector method) evaluates velocity at both the start and predicted end position, then averages them for a more accurate step."
    },
    {
      question: "What does NFE mean in the context of ODE solvers?",
      options: [
        "Neural Function Estimation",
        "Number of Function Evaluations (model calls)",
        "Normalized Flow Error",
        "Network Forward Execution"
      ],
      correct: 1,
      explanation: "NFE = Number of Function Evaluations, counting how many times we call the neural network. More NFE = more compute but potentially better quality."
    },
    {
      question: "In the Karras noise schedule, what does the ρ parameter control?",
      options: [
        "The total number of steps",
        "The curvature of the schedule (how sigmas are spaced)",
        "The final noise level",
        "The learning rate"
      ],
      correct: 1,
      explanation: "ρ (rho) controls the spacing of sigma values. Higher ρ means more steps at low noise levels. ρ=7 is commonly used."
    },
    {
      question: "What is the relationship between sigma (σ) and SNR?",
      options: [
        "SNR = σ",
        "SNR = 1/σ",
        "SNR = (1-σ²)/σ²",
        "SNR = σ²"
      ],
      correct: 2,
      explanation: "SNR (Signal-to-Noise Ratio) = α²/σ² where α = √(1-σ²). Higher σ means more noise, lower SNR."
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

  // Calculate logit-normal value
  const logitNormalPDF = (t, mu, sigma) => {
    if (t <= 0 || t >= 1) return 0;
    const logit = Math.log(t / (1 - t));
    const exponent = -((logit - mu) ** 2) / (2 * sigma ** 2);
    const normalization = 1 / (sigma * Math.sqrt(2 * Math.PI));
    const jacobian = 1 / (t * (1 - t));
    return normalization * Math.exp(exponent) * jacobian;
  };

  // Calculate Euler step
  const eulerStep = (x, v, dt) => x + v * dt;

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          Practice Lab: <span className="text-fuchsia-400">Test Your Knowledge</span>
        </h2>
        <p className="text-gray-400">Quiz yourself and explore interactive calculations</p>
      </div>

      {/* Mode Selector */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setMode('quiz')}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            mode === 'quiz'
              ? 'bg-fuchsia-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Brain size={20} />
          Knowledge Quiz
        </button>
        <button
          onClick={() => setMode('calculator')}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            mode === 'calculator'
              ? 'bg-fuchsia-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Calculator size={20} />
          Interactive Calculator
        </button>
        <button
          onClick={() => setMode('code')}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
            mode === 'code'
              ? 'bg-fuchsia-600 text-white'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          <Code size={20} />
          Code Exercises
        </button>
      </div>

      {/* Quiz Mode */}
      {mode === 'quiz' && (
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          {/* Score */}
          <div className="flex justify-between items-center mb-6">
            <span className="text-gray-400">
              Question {currentQuestion + 1} of {questions.length}
            </span>
            <span className="text-fuchsia-400 font-bold">
              Score: {score}/{answered}
            </span>
          </div>

          {/* Progress */}
          <div className="h-2 bg-white/10 rounded-full mb-6 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-fuchsia-500 to-purple-500 transition-all duration-300"
              style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
            />
          </div>

          {/* Question */}
          <div className="mb-6">
            <h3 className="text-xl font-bold mb-4">{questions[currentQuestion].question}</h3>
            <div className="space-y-3">
              {questions[currentQuestion].options.map((option, index) => {
                const isCorrect = index === questions[currentQuestion].correct;
                const isSelected = selectedAnswer === index;

                let buttonClass = 'w-full text-left p-4 rounded-xl border transition-all ';
                if (showResult) {
                  if (isCorrect) {
                    buttonClass += 'border-green-500 bg-green-500/20 text-green-300';
                  } else if (isSelected) {
                    buttonClass += 'border-red-500 bg-red-500/20 text-red-300';
                  } else {
                    buttonClass += 'border-white/10 bg-white/5 text-gray-500';
                  }
                } else {
                  buttonClass += 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-fuchsia-500/50';
                }

                return (
                  <button
                    key={index}
                    onClick={() => handleAnswer(index)}
                    className={buttonClass}
                    disabled={showResult}
                  >
                    <div className="flex items-center gap-3">
                      {showResult && isCorrect && <CheckCircle className="text-green-400" size={20} />}
                      {showResult && isSelected && !isCorrect && <XCircle className="text-red-400" size={20} />}
                      <span>{option}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Explanation */}
          {showResult && (
            <div className="bg-white/5 rounded-xl p-4 mb-6">
              <p className="text-gray-300">
                <strong className="text-fuchsia-400">Explanation:</strong>{' '}
                {questions[currentQuestion].explanation}
              </p>
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between">
            <button
              onClick={resetQuiz}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
            >
              <RotateCcw size={18} />
              Reset Quiz
            </button>
            {showResult && currentQuestion < questions.length - 1 && (
              <button
                onClick={nextQuestion}
                className="px-6 py-2 bg-fuchsia-600 hover:bg-fuchsia-700 rounded-lg transition-colors"
              >
                Next Question →
              </button>
            )}
            {showResult && currentQuestion === questions.length - 1 && (
              <div className="text-lg font-bold text-fuchsia-400">
                Quiz Complete! Final Score: {score}/{questions.length}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Calculator Mode */}
      {mode === 'calculator' && (
        <div className="space-y-6">
          {/* Logit-Normal Calculator */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">Logit-Normal PDF Calculator</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400 block mb-2">t = {calcT.toFixed(2)}</label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.99"
                    step="0.01"
                    value={calcT}
                    onChange={(e) => setCalcT(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 block mb-2">μ = {calcMean.toFixed(1)}</label>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={calcMean}
                    onChange={(e) => setCalcMean(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 block mb-2">σ = {calcStd.toFixed(1)}</label>
                  <input
                    type="range"
                    min="0.3"
                    max="2"
                    step="0.1"
                    value={calcStd}
                    onChange={(e) => setCalcStd(parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <h4 className="font-bold text-fuchsia-300 mb-3">Results</h4>
                <div className="space-y-2 font-mono text-sm">
                  <p>logit(t) = log({calcT.toFixed(2)} / {(1-calcT).toFixed(2)}) = <span className="text-fuchsia-400">{Math.log(calcT / (1 - calcT)).toFixed(4)}</span></p>
                  <p>PDF(t) = <span className="text-fuchsia-400">{logitNormalPDF(calcT, calcMean, calcStd).toFixed(4)}</span></p>
                  <p>Mode ≈ <span className="text-purple-400">{(1 / (1 + Math.exp(-calcMean))).toFixed(3)}</span></p>
                </div>
              </div>
            </div>
          </div>

          {/* Euler Step Calculator */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">Euler Step Calculator</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white/5 rounded-xl p-4">
                <h4 className="text-sm text-gray-400 mb-2">Current Position (x)</h4>
                <input
                  type="number"
                  defaultValue="0"
                  className="w-full bg-black/30 rounded px-3 py-2 text-white"
                  id="euler-x"
                />
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <h4 className="text-sm text-gray-400 mb-2">Velocity (v)</h4>
                <input
                  type="number"
                  defaultValue="1"
                  className="w-full bg-black/30 rounded px-3 py-2 text-white"
                  id="euler-v"
                />
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <h4 className="text-sm text-gray-400 mb-2">Step Size (dt)</h4>
                <input
                  type="number"
                  defaultValue="0.1"
                  step="0.01"
                  className="w-full bg-black/30 rounded px-3 py-2 text-white"
                  id="euler-dt"
                />
              </div>
            </div>
            <div className="mt-4 p-4 bg-fuchsia-500/20 rounded-xl text-center">
              <p className="text-gray-400">x_new = x + v × dt</p>
              <p className="text-2xl font-bold text-fuchsia-400 mt-2">
                = 0 + 1 × 0.1 = 0.1
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Code Exercises Mode */}
      {mode === 'code' && (
        <div className="space-y-6">
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">Exercise 1: Implement Linear Interpolation</h3>
            <p className="text-gray-400 mb-4">
              Complete the function to perform linear interpolation between noise x₀ and data x₁:
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-300">{`fn interpolate(x0: f32, x1: f32, t: f32) -> f32 {
    // TODO: Return the interpolated value
    // x_t = (1-t)*x0 + t*x1
    ???
}`}</pre>
            </div>
            <details className="mt-4">
              <summary className="text-fuchsia-400 cursor-pointer hover:text-fuchsia-300">Show Solution</summary>
              <div className="bg-black/50 rounded-lg p-4 font-mono text-sm mt-2">
                <pre className="text-green-400">{`fn interpolate(x0: f32, x1: f32, t: f32) -> f32 {
    (1.0 - t) * x0 + t * x1
}`}</pre>
              </div>
            </details>
          </div>

          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">Exercise 2: Implement Euler Sampler</h3>
            <p className="text-gray-400 mb-4">
              Complete the Euler sampling loop:
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-300">{`fn euler_sample(model: &Model, num_steps: usize) -> Tensor {
    let mut x = sample_noise();  // Start from noise
    let dt = 1.0 / num_steps as f32;
    
    for i in 0..num_steps {
        let t = i as f32 / num_steps as f32;
        let v = model.predict_velocity(&x, t);
        // TODO: Update x using Euler step
        ???
    }
    x
}`}</pre>
            </div>
            <details className="mt-4">
              <summary className="text-fuchsia-400 cursor-pointer hover:text-fuchsia-300">Show Solution</summary>
              <div className="bg-black/50 rounded-lg p-4 font-mono text-sm mt-2">
                <pre className="text-green-400">{`fn euler_sample(model: &Model, num_steps: usize) -> Tensor {
    let mut x = sample_noise();
    let dt = 1.0 / num_steps as f32;
    
    for i in 0..num_steps {
        let t = i as f32 / num_steps as f32;
        let v = model.predict_velocity(&x, t);
        x = x + v * dt;  // Euler step!
    }
    x
}`}</pre>
              </div>
            </details>
          </div>

          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <h3 className="text-xl font-bold mb-4">Exercise 3: Sample from Logit-Normal</h3>
            <p className="text-gray-400 mb-4">
              Implement logit-normal sampling for timesteps:
            </p>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm">
              <pre className="text-gray-300">{`fn sample_logit_normal(mean: f32, std: f32) -> f32 {
    let z: f32 = /* sample from N(0,1) */;
    // TODO: Apply logit-normal transform
    // 1. Compute logit = mean + std * z
    // 2. Apply sigmoid to get t in (0, 1)
    ???
}`}</pre>
            </div>
            <details className="mt-4">
              <summary className="text-fuchsia-400 cursor-pointer hover:text-fuchsia-300">Show Solution</summary>
              <div className="bg-black/50 rounded-lg p-4 font-mono text-sm mt-2">
                <pre className="text-green-400">{`fn sample_logit_normal(mean: f32, std: f32) -> f32 {
    let z = randn();  // Standard normal sample
    let logit = mean + std * z;
    1.0 / (1.0 + (-logit).exp())  // Sigmoid
}`}</pre>
              </div>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
