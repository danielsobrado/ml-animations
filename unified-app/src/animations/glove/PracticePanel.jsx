import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Trophy, HelpCircle, ArrowRight } from 'lucide-react';

export default function PracticePanel() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answered, setAnswered] = useState([]);

  const questions = [
    {
      question: 'What does GloVe stand for?',
      options: [
        'Global Vectors for Word Representation',
        'Graphical Language Output Vector Encoding',
        'Generalized Linear Output Vectors',
        'Global Language Vector Embeddings'
      ],
      correct: 0,
      explanation: 'GloVe stands for Global Vectors for Word Representation. It emphasizes the use of global word-word co-occurrence statistics.'
    },
    {
      question: 'What is the co-occurrence matrix X in GloVe?',
      options: [
        'A matrix where X_ij is the count of word j appearing in context of word i',
        'A matrix of word similarities',
        'A matrix of TF-IDF scores',
        'A matrix of one-hot encoded words'
      ],
      correct: 0,
      explanation: 'The co-occurrence matrix X_ij counts how many times word j appears within a context window of word i across the entire corpus.'
    },
    {
      question: 'Why does GloVe use a weighting function f(X_ij)?',
      options: [
        'To speed up training',
        'To prevent frequent word pairs from dominating the objective',
        'To normalize the embeddings',
        'To handle out-of-vocabulary words'
      ],
      correct: 1,
      explanation: 'The weighting function f(X_ij) prevents very common word pairs (like "the, of") from dominating the loss. It caps at x_max and uses α=0.75 for smoothing.'
    },
    {
      question: 'What is the key insight behind GloVe\'s objective function?',
      options: [
        'Word similarity is based on word length',
        'Probability ratios encode semantic relationships',
        'Words should have orthogonal vectors',
        'Frequent words are more important'
      ],
      correct: 1,
      explanation: 'GloVe\'s key insight is that the ratio P(k|ice)/P(k|steam) reveals whether word k is more related to "ice" or "steam", capturing semantic relationships.'
    },
    {
      question: 'How does GloVe differ from Word2Vec in training approach?',
      options: [
        'GloVe uses neural networks, Word2Vec uses matrix factorization',
        'GloVe pre-computes global statistics, Word2Vec uses local windows online',
        'GloVe is faster to train',
        'GloVe requires labeled data'
      ],
      correct: 1,
      explanation: 'GloVe pre-computes a co-occurrence matrix from the entire corpus before training, while Word2Vec trains incrementally using local context windows and SGD.'
    },
    {
      question: 'What is the typical value of x_max in the weighting function?',
      options: [
        '10',
        '50',
        '100',
        '1000'
      ],
      correct: 2,
      explanation: 'The authors recommend x_max = 100, meaning co-occurrences above 100 are capped at weight 1.0 to prevent very common pairs from dominating.'
    },
    {
      question: 'What is the final word embedding in GloVe?',
      options: [
        'Just the word vector W',
        'Just the context vector W̃',
        'W + W̃ (sum of both)',
        'W × W̃ (product of both)'
      ],
      correct: 2,
      explanation: 'Since W and W̃ are symmetric in GloVe\'s objective, the final embedding is W + W̃, which gives slightly better performance than using either alone.'
    },
    {
      question: 'Which organization released GloVe?',
      options: [
        'Google',
        'Stanford NLP Group',
        'Facebook AI',
        'OpenAI'
      ],
      correct: 1,
      explanation: 'GloVe was released by the Stanford NLP Group in 2014, authored by Jeffrey Pennington, Richard Socher, and Christopher Manning.'
    },
    {
      question: 'What does the α parameter (typically 0.75) control?',
      options: [
        'Learning rate',
        'Embedding dimension',
        'Smoothing of the weighting function',
        'Context window size'
      ],
      correct: 2,
      explanation: 'α = 0.75 is the power in f(x) = (x/x_max)^α, which smooths the weighting so that rare co-occurrences still contribute meaningfully.'
    },
    {
      question: 'Which statement about GloVe is TRUE?',
      options: [
        'GloVe can be trained incrementally on new data',
        'GloVe uses negative sampling like Word2Vec',
        'GloVe training only considers non-zero co-occurrence entries',
        'GloVe produces sparse word vectors'
      ],
      correct: 2,
      explanation: 'GloVe only trains on observed (non-zero) co-occurrences. Unlike Word2Vec\'s negative sampling, GloVe uses weighted least squares on the sparse matrix.'
    }
  ];

  const handleAnswer = (answerIndex) => {
    if (showResult) return;
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    
    const isCorrect = answerIndex === questions[currentQuestion].correct;
    if (isCorrect) {
      setScore(prev => prev + 1);
    }
    setAnswered([...answered, { question: currentQuestion, correct: isCorrect }]);
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
    setAnswered([]);
  };

  const isQuizComplete = answered.length === questions.length;

  return (
    <div className="space-y-6 pb-20">
      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Practice</span> Quiz
        </h2>
        <p className="text-gray-400">
          Test your understanding of GloVe
        </p>
      </div>

      {/* Progress */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Progress</span>
          <span className="text-sm text-violet-400">{answered.length} / {questions.length}</span>
        </div>
        <div className="flex gap-1">
          {questions.map((_, i) => (
            <div
              key={i}
              className={`h-2 flex-1 rounded-full transition-all ${
                answered[i] !== undefined
                  ? answered[i].correct
                    ? 'bg-green-500'
                    : 'bg-red-500'
                  : i === currentQuestion
                  ? 'bg-violet-500'
                  : 'bg-white/10'
              }`}
            />
          ))}
        </div>
        <div className="flex justify-between items-center mt-2">
          <span className="text-sm text-gray-400">Score</span>
          <span className="text-sm text-green-400">{score} correct</span>
        </div>
      </div>

      {/* Quiz Complete */}
      {isQuizComplete ? (
        <div className="bg-gradient-to-r from-violet-900/30 to-cyan-900/30 rounded-2xl p-8 border border-violet-500/30 text-center">
          <Trophy size={64} className="mx-auto text-yellow-400 mb-4" />
          <h3 className="text-2xl font-bold text-white mb-2">Quiz Complete!</h3>
          <p className="text-4xl font-bold text-violet-400 mb-4">
            {score} / {questions.length}
          </p>
          <p className="text-gray-400 mb-6">
            {score === questions.length 
              ? 'Perfect score! You\'ve mastered GloVe!' 
              : score >= questions.length * 0.7
              ? 'Great job! You have a solid understanding of GloVe.'
              : 'Keep practicing! Review the concepts and try again.'}
          </p>
          <button
            onClick={resetQuiz}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-violet-600 hover:bg-violet-700 rounded-lg transition-colors"
          >
            <RotateCcw size={18} />
            Try Again
          </button>
        </div>
      ) : (
        <>
          {/* Question Card */}
          <div className="bg-black/30 rounded-2xl p-6 border border-white/10">
            <div className="flex items-center gap-2 mb-4">
              <span className="px-3 py-1 bg-violet-600 rounded-full text-sm">
                Question {currentQuestion + 1}
              </span>
            </div>
            
            <h3 className="text-xl font-medium text-white mb-6">
              {questions[currentQuestion].question}
            </h3>

            <div className="grid gap-3">
              {questions[currentQuestion].options.map((option, i) => (
                <button
                  key={i}
                  onClick={() => handleAnswer(i)}
                  disabled={showResult}
                  className={`p-4 rounded-lg text-left transition-all border ${
                    showResult
                      ? i === questions[currentQuestion].correct
                        ? 'bg-green-900/30 border-green-500 text-green-400'
                        : i === selectedAnswer
                        ? 'bg-red-900/30 border-red-500 text-red-400'
                        : 'bg-white/5 border-white/10 text-gray-500'
                      : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20 text-white'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      showResult
                        ? i === questions[currentQuestion].correct
                          ? 'bg-green-500 text-black'
                          : i === selectedAnswer
                          ? 'bg-red-500 text-white'
                          : 'bg-white/10'
                        : 'bg-white/10'
                    }`}>
                      {showResult && i === questions[currentQuestion].correct ? (
                        <CheckCircle size={18} />
                      ) : showResult && i === selectedAnswer ? (
                        <XCircle size={18} />
                      ) : (
                        String.fromCharCode(65 + i)
                      )}
                    </span>
                    <span>{option}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Explanation */}
          {showResult && (
            <div className={`rounded-xl p-4 border ${
              selectedAnswer === questions[currentQuestion].correct
                ? 'bg-green-900/20 border-green-500/30'
                : 'bg-red-900/20 border-red-500/30'
            }`}>
              <div className="flex items-start gap-3">
                {selectedAnswer === questions[currentQuestion].correct ? (
                  <CheckCircle className="text-green-400 mt-1 flex-shrink-0" size={20} />
                ) : (
                  <XCircle className="text-red-400 mt-1 flex-shrink-0" size={20} />
                )}
                <div>
                  <p className={`font-medium ${
                    selectedAnswer === questions[currentQuestion].correct
                      ? 'text-green-400'
                      : 'text-red-400'
                  }`}>
                    {selectedAnswer === questions[currentQuestion].correct
                      ? 'Correct!'
                      : 'Not quite right'}
                  </p>
                  <p className="text-gray-300 mt-1 text-sm">
                    {questions[currentQuestion].explanation}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Next Button */}
          {showResult && currentQuestion < questions.length - 1 && (
            <div className="flex justify-center">
              <button
                onClick={nextQuestion}
                className="flex items-center gap-2 px-6 py-3 bg-violet-600 hover:bg-violet-700 rounded-lg transition-colors"
              >
                Next Question
                <ArrowRight size={18} />
              </button>
            </div>
          )}
        </>
      )}

      {/* Quick Reference */}
      <div className="bg-gradient-to-r from-violet-900/20 to-cyan-900/20 rounded-xl p-6 border border-violet-500/30">
        <h4 className="flex items-center gap-2 font-bold text-violet-400 mb-4">
          <HelpCircle size={18} />
          Quick Reference
        </h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-violet-400 font-medium mb-1">GloVe Objective</p>
            <p className="text-gray-400 font-mono text-xs">J = Σ f(X)(w·w̃ + b - log X)²</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-cyan-400 font-medium mb-1">Weighting Function</p>
            <p className="text-gray-400 font-mono text-xs">f(x) = (x/x_max)^0.75 if x &lt; 100</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Final Embedding</p>
            <p className="text-gray-400">W + W̃ (word + context)</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">Key Insight</p>
            <p className="text-gray-400">P(k|i) / P(k|j) encodes meaning</p>
          </div>
        </div>
      </div>
    </div>
  );
}
