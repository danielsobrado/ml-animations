import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Calculator, Trophy, Brain, Lightbulb } from 'lucide-react';

export default function PracticePanel() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [quizComplete, setQuizComplete] = useState(false);

  // Parameter Calculator State
  const [calcConfig, setCalcConfig] = useState({
    layers: 12,
    hidden: 768,
    heads: 12,
    intermediate: 3072,
    vocab: 30522,
    maxPos: 512
  });

  const questions = [
    {
      question: "What does BERT stand for?",
      options: [
        "Bidirectional Encoder Representations from Transformers",
        "Binary Encoded Recurrent Transformer",
        "Baseline Encoder for Reading Text",
        "Bidirectional Entity Recognition Transformer"
      ],
      correct: 0,
      explanation: "BERT = Bidirectional Encoder Representations from Transformers. The key insight is 'bidirectional' - BERT looks at context from both directions."
    },
    {
      question: "What percentage of tokens does BERT mask during pre-training?",
      options: ["5%", "10%", "15%", "20%"],
      correct: 2,
      explanation: "BERT masks 15% of tokens: 80% become [MASK], 10% become random tokens, and 10% stay unchanged."
    },
    {
      question: "What are the two pre-training objectives of BERT?",
      options: [
        "Next Word Prediction & Sentence Classification",
        "Masked Language Model & Next Sentence Prediction",
        "Token Classification & Text Generation",
        "Autoencoding & Autoregressive Modeling"
      ],
      correct: 1,
      explanation: "BERT uses MLM (Masked Language Model) to learn token representations and NSP (Next Sentence Prediction) to learn sentence relationships."
    },
    {
      question: "Which special token does BERT use for classification tasks?",
      options: ["[MASK]", "[SEP]", "[CLS]", "[PAD]"],
      correct: 2,
      explanation: "[CLS] (classification) token is added at the start. Its final hidden state is used for classification tasks."
    },
    {
      question: "How many attention heads does BERT-Base have per layer?",
      options: ["8", "12", "16", "24"],
      correct: 1,
      explanation: "BERT-Base has 12 attention heads with 64 dimensions each (12 × 64 = 768 total hidden size)."
    },
    {
      question: "What is the maximum sequence length BERT can process?",
      options: ["128 tokens", "256 tokens", "512 tokens", "1024 tokens"],
      correct: 2,
      explanation: "BERT has learned position embeddings for up to 512 positions, limiting its maximum sequence length."
    },
    {
      question: "How does BERT differ from GPT in terms of attention?",
      options: [
        "BERT uses masked attention, GPT uses full attention",
        "BERT uses bidirectional attention, GPT uses causal (left-to-right) attention",
        "BERT uses no attention, GPT uses self-attention",
        "They use the same attention mechanism"
      ],
      correct: 1,
      explanation: "BERT sees all tokens (bidirectional), while GPT can only attend to previous tokens (causal/autoregressive)."
    },
    {
      question: "What are the three components summed to create BERT's input embeddings?",
      options: [
        "Word, Character, Sentence embeddings",
        "Token, Position, Segment embeddings",
        "Input, Output, Hidden embeddings",
        "Query, Key, Value embeddings"
      ],
      correct: 1,
      explanation: "Input = Token embedding (word) + Position embedding (location) + Segment embedding (sentence A or B)."
    },
    {
      question: "How many encoder layers does BERT-Large have?",
      options: ["6", "12", "24", "48"],
      correct: 2,
      explanation: "BERT-Large has 24 layers, 16 attention heads, and 1024 hidden dimensions (340M parameters total)."
    },
    {
      question: "Which activation function does BERT use in its feed-forward layers?",
      options: ["ReLU", "Sigmoid", "GELU", "Tanh"],
      correct: 2,
      explanation: "BERT uses GELU (Gaussian Error Linear Unit), which is smoother than ReLU and works better for NLP tasks."
    }
  ];

  const handleAnswer = (answerIndex) => {
    if (showResult) return;
    setSelectedAnswer(answerIndex);
  };

  const checkAnswer = () => {
    if (selectedAnswer === null) return;
    setShowResult(true);
    if (selectedAnswer === questions[currentQuestion].correct) {
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

  // Calculate BERT parameters
  const calculateParams = () => {
    const { layers, hidden, heads, intermediate, vocab, maxPos } = calcConfig;
    
    // Embedding parameters
    const tokenEmbed = vocab * hidden;
    const posEmbed = maxPos * hidden;
    const segmentEmbed = 2 * hidden;
    const embedLayerNorm = 2 * hidden; // gamma and beta
    const totalEmbed = tokenEmbed + posEmbed + segmentEmbed + embedLayerNorm;

    // Per-layer parameters
    const qkvWeights = 3 * hidden * hidden; // Q, K, V projections
    const qkvBias = 3 * hidden;
    const outputProj = hidden * hidden;
    const outputBias = hidden;
    const attentionLN = 2 * hidden;
    
    const ffnUp = hidden * intermediate;
    const ffnUpBias = intermediate;
    const ffnDown = intermediate * hidden;
    const ffnDownBias = hidden;
    const ffnLN = 2 * hidden;
    
    const perLayer = qkvWeights + qkvBias + outputProj + outputBias + attentionLN +
                     ffnUp + ffnUpBias + ffnDown + ffnDownBias + ffnLN;
    
    const totalLayers = perLayer * layers;

    // Pooler
    const pooler = hidden * hidden + hidden;

    const total = totalEmbed + totalLayers + pooler;

    return {
      embeddings: totalEmbed,
      perLayer,
      allLayers: totalLayers,
      pooler,
      total
    };
  };

  const params = calculateParams();

  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toString();
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="text-yellow-400">Practice</span> Lab
        </h2>
        <p className="text-gray-400">
          Test your BERT knowledge and explore model parameters
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Quiz Section */}
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Brain className="text-yellow-400" />
            BERT Knowledge Quiz
          </h3>

          {!quizComplete ? (
            <>
              {/* Progress */}
              <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Question {currentQuestion + 1} of {questions.length}</span>
                  <span>Score: {score}/{currentQuestion + (showResult ? 1 : 0)}</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-yellow-500 transition-all duration-300"
                    style={{ width: `${((currentQuestion + (showResult ? 1 : 0)) / questions.length) * 100}%` }}
                  />
                </div>
              </div>

              {/* Question */}
              <div className="bg-yellow-900/20 rounded-xl p-4 border border-yellow-500/30 mb-4">
                <p className="font-medium text-lg">{questions[currentQuestion].question}</p>
              </div>

              {/* Options */}
              <div className="space-y-2 mb-4">
                {questions[currentQuestion].options.map((option, i) => (
                  <button
                    key={i}
                    onClick={() => handleAnswer(i)}
                    disabled={showResult}
                    className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                      showResult
                        ? i === questions[currentQuestion].correct
                          ? 'bg-green-900/30 border-green-500 text-green-300'
                          : i === selectedAnswer
                          ? 'bg-red-900/30 border-red-500 text-red-300'
                          : 'bg-gray-800/30 border-gray-600 text-gray-500'
                        : selectedAnswer === i
                        ? 'bg-yellow-900/30 border-yellow-500'
                        : 'bg-gray-800/30 border-gray-600 hover:border-gray-400'
                    }`}
                  >
                    <span className="font-mono mr-2">{String.fromCharCode(65 + i)}.</span>
                    {option}
                    {showResult && i === questions[currentQuestion].correct && (
                      <CheckCircle className="inline ml-2 text-green-400" size={18} />
                    )}
                    {showResult && i === selectedAnswer && i !== questions[currentQuestion].correct && (
                      <XCircle className="inline ml-2 text-red-400" size={18} />
                    )}
                  </button>
                ))}
              </div>

              {/* Explanation */}
              {showResult && (
                <div className="bg-blue-900/20 rounded-xl p-4 border border-blue-500/30 mb-4">
                  <div className="flex items-start gap-2">
                    <Lightbulb className="text-blue-400 shrink-0 mt-0.5" size={18} />
                    <p className="text-sm text-gray-300">{questions[currentQuestion].explanation}</p>
                  </div>
                </div>
              )}

              {/* Buttons */}
              <div className="flex gap-3">
                {!showResult ? (
                  <button
                    onClick={checkAnswer}
                    disabled={selectedAnswer === null}
                    className="flex-1 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
                  >
                    Check Answer
                  </button>
                ) : (
                  <button
                    onClick={nextQuestion}
                    className="flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                  >
                    {currentQuestion < questions.length - 1 ? 'Next Question' : 'See Results'}
                  </button>
                )}
              </div>
            </>
          ) : (
            /* Quiz Complete */
            <div className="text-center py-8">
              <Trophy className="mx-auto text-yellow-400 mb-4" size={64} />
              <h4 className="text-2xl font-bold mb-2">Quiz Complete!</h4>
              <p className="text-4xl font-bold text-yellow-400 mb-2">
                {score} / {questions.length}
              </p>
              <p className="text-gray-400 mb-4">
                {score === questions.length ? "Perfect score! You're a BERT expert! 🎉" :
                 score >= questions.length * 0.8 ? "Great job! You know BERT well! 👏" :
                 score >= questions.length * 0.6 ? "Good effort! Keep learning! 📚" :
                 "Keep practicing! Review the materials. 💪"}
              </p>
              <button
                onClick={resetQuiz}
                className="flex items-center gap-2 mx-auto px-6 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
              >
                <RotateCcw size={18} />
                Try Again
              </button>
            </div>
          )}
        </div>

        {/* Parameter Calculator */}
        <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Calculator className="text-purple-400" />
            BERT Parameter Calculator
          </h3>

          {/* Config Inputs */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            <div>
              <label className="text-xs text-gray-400">Layers (L)</label>
              <input
                type="number"
                value={calcConfig.layers}
                onChange={(e) => setCalcConfig({...calcConfig, layers: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Hidden Size (H)</label>
              <input
                type="number"
                value={calcConfig.hidden}
                onChange={(e) => setCalcConfig({...calcConfig, hidden: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Attention Heads</label>
              <input
                type="number"
                value={calcConfig.heads}
                onChange={(e) => setCalcConfig({...calcConfig, heads: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">FFN Intermediate</label>
              <input
                type="number"
                value={calcConfig.intermediate}
                onChange={(e) => setCalcConfig({...calcConfig, intermediate: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Vocab Size</label>
              <input
                type="number"
                value={calcConfig.vocab}
                onChange={(e) => setCalcConfig({...calcConfig, vocab: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Max Positions</label>
              <input
                type="number"
                value={calcConfig.maxPos}
                onChange={(e) => setCalcConfig({...calcConfig, maxPos: parseInt(e.target.value) || 0})}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              />
            </div>
          </div>

          {/* Presets */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setCalcConfig({ layers: 12, hidden: 768, heads: 12, intermediate: 3072, vocab: 30522, maxPos: 512 })}
              className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600/50 rounded text-sm"
            >
              BERT-Base
            </button>
            <button
              onClick={() => setCalcConfig({ layers: 24, hidden: 1024, heads: 16, intermediate: 4096, vocab: 30522, maxPos: 512 })}
              className="px-3 py-1 bg-purple-600/30 hover:bg-purple-600/50 rounded text-sm"
            >
              BERT-Large
            </button>
            <button
              onClick={() => setCalcConfig({ layers: 6, hidden: 768, heads: 12, intermediate: 3072, vocab: 30522, maxPos: 512 })}
              className="px-3 py-1 bg-green-600/30 hover:bg-green-600/50 rounded text-sm"
            >
              DistilBERT
            </button>
          </div>

          {/* Results */}
          <div className="space-y-2">
            <div className="flex justify-between items-center p-2 bg-gray-800/50 rounded">
              <span className="text-sm text-gray-400">Embedding Parameters</span>
              <span className="font-mono text-blue-400">{formatNumber(params.embeddings)}</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-gray-800/50 rounded">
              <span className="text-sm text-gray-400">Per Layer</span>
              <span className="font-mono text-green-400">{formatNumber(params.perLayer)}</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-gray-800/50 rounded">
              <span className="text-sm text-gray-400">All {calcConfig.layers} Layers</span>
              <span className="font-mono text-purple-400">{formatNumber(params.allLayers)}</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-gray-800/50 rounded">
              <span className="text-sm text-gray-400">Pooler</span>
              <span className="font-mono text-orange-400">{formatNumber(params.pooler)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-yellow-900/30 rounded-lg border border-yellow-500/30">
              <span className="font-bold text-yellow-400">Total Parameters</span>
              <span className="font-mono text-2xl text-yellow-400">{formatNumber(params.total)}</span>
            </div>
          </div>

          {/* Breakdown Formula */}
          <div className="mt-4 p-3 bg-black/30 rounded-lg text-xs text-gray-400">
            <p className="font-medium text-gray-300 mb-1">Parameter Breakdown:</p>
            <p>Embeddings: V×H + P×H + 2×H + 2×H</p>
            <p>Attention: 4×H² + 4×H + 2×H</p>
            <p>FFN: H×I + I + I×H + H + 2×H</p>
            <p className="mt-1 text-gray-500">V=vocab, H=hidden, P=positions, I=intermediate</p>
          </div>
        </div>
      </div>

      {/* Key Takeaways */}
      <div className="bg-gradient-to-r from-yellow-900/20 to-orange-900/20 rounded-2xl p-6 border border-yellow-500/30">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Lightbulb className="text-yellow-400" />
          Key BERT Takeaways
        </h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">🔄 Bidirectional</p>
            <p className="text-sm text-gray-300">Unlike GPT, BERT sees both left and right context for each token.</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">🎭 Masked LM</p>
            <p className="text-sm text-gray-300">Pre-training by predicting randomly masked tokens (15%).</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">📑 NSP Task</p>
            <p className="text-sm text-gray-300">Learns sentence relationships via Next Sentence Prediction.</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">🎯 [CLS] Token</p>
            <p className="text-sm text-gray-300">Special token for sequence-level classification tasks.</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">🔧 Fine-Tuning</p>
            <p className="text-sm text-gray-300">Add task head + train entire model on downstream tasks.</p>
          </div>
          <div className="bg-black/30 rounded-lg p-4">
            <p className="text-yellow-400 font-bold mb-1">📊 GLUE Champion</p>
            <p className="text-sm text-gray-300">Achieved SOTA on 11 NLP tasks when released in 2018.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
