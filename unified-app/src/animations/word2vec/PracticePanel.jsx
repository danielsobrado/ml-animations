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
      type: 'multiple',
      question: 'What is the main idea behind Word2Vec?',
      options: [
        'Words that appear in similar contexts have similar meanings',
        'Words should be represented as one-hot vectors',
        'All words are equally different from each other',
        'Word order is the most important feature'
      ],
      correct: 0,
      explanation: 'Word2Vec is based on the distributional hypothesis: "You shall know a word by the company it keeps." Words appearing in similar contexts tend to have similar vector representations.'
    },
    {
      type: 'multiple',
      question: 'In Skip-gram, what is the input and what is the output?',
      options: [
        'Input: context words, Output: center word',
        'Input: center word, Output: context words',
        'Input: sentence, Output: document vector',
        'Input: one-hot vector, Output: TF-IDF vector'
      ],
      correct: 1,
      explanation: 'Skip-gram takes a center word as input and tries to predict the surrounding context words. This is the opposite of CBOW.'
    },
    {
      type: 'multiple',
      question: 'What is the purpose of negative sampling?',
      options: [
        'To increase vocabulary size',
        'To make training faster by avoiding full softmax',
        'To generate negative reviews',
        'To remove stop words'
      ],
      correct: 1,
      explanation: 'Negative sampling converts the expensive softmax over the entire vocabulary into a binary classification task, making training much faster (O(V) → O(k)).'
    },
    {
      type: 'multiple',
      question: 'What does "king - man + woman ≈ queen" demonstrate?',
      options: [
        'Word2Vec cannot do arithmetic',
        'Word vectors capture semantic relationships',
        'All words have the same vector',
        'One-hot encoding is better than embeddings'
      ],
      correct: 1,
      explanation: 'This famous analogy shows that Word2Vec vectors capture semantic relationships. The gender relationship is encoded as a consistent direction in the embedding space.'
    },
    {
      type: 'multiple',
      question: 'Which architecture is typically faster to train?',
      options: [
        'Skip-gram',
        'CBOW',
        'Both are equally fast',
        'Neither uses neural networks'
      ],
      correct: 1,
      explanation: 'CBOW is typically faster because it makes one prediction per context window, while Skip-gram makes multiple predictions (one for each context word).'
    },
    {
      type: 'multiple',
      question: 'What is the typical dimensionality of Word2Vec embeddings?',
      options: [
        '10-50 dimensions',
        '100-300 dimensions',
        '1000-10000 dimensions',
        'Vocabulary size'
      ],
      correct: 1,
      explanation: 'Word2Vec embeddings are typically 100-300 dimensions. This is much smaller than one-hot vectors (vocabulary size) but rich enough to capture semantic relationships.'
    },
    {
      type: 'multiple',
      question: 'Which metric is commonly used to measure word similarity in embedding space?',
      options: [
        'Euclidean distance',
        'Manhattan distance',
        'Cosine similarity',
        'Hamming distance'
      ],
      correct: 2,
      explanation: 'Cosine similarity measures the angle between vectors, which captures semantic similarity regardless of vector magnitude. Values range from -1 to 1.'
    },
    {
      type: 'multiple',
      question: 'What is a major limitation of Word2Vec?',
      options: [
        'It uses too much memory',
        'Each word has only one vector (no context awareness)',
        'It can only work with English',
        'It requires labeled training data'
      ],
      correct: 1,
      explanation: 'Word2Vec assigns a single static vector to each word, so "bank" has the same vector whether it means riverbank or financial bank. This is solved by contextual embeddings like BERT.'
    },
    {
      type: 'multiple',
      question: 'In negative sampling, how are negative samples typically selected?',
      options: [
        'Completely randomly',
        'Based on word frequency raised to 0.75 power',
        'Only stop words',
        'Words with opposite meaning'
      ],
      correct: 1,
      explanation: 'Negative samples are drawn from a distribution proportional to word frequency^0.75. The 0.75 power smooths the distribution, giving rare words more chance to be selected.'
    },
    {
      type: 'multiple',
      question: 'What determines the "context" in Word2Vec training?',
      options: [
        'The entire document',
        'Words within a fixed window size',
        'Only the immediately adjacent words',
        'All words in the corpus'
      ],
      correct: 1,
      explanation: 'Context is defined by a window size parameter (typically 5-10). Words within this window of the center word are considered context words for training.'
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
          <span className="text-pink-400">Practice</span> Quiz
        </h2>
        <p className="text-gray-400">
          Test your understanding of Word2Vec
        </p>
      </div>

      {/* Progress */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Progress</span>
          <span className="text-sm text-pink-400">{answered.length} / {questions.length}</span>
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
                  ? 'bg-pink-500'
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
        <div className="bg-gradient-to-r from-pink-900/30 to-purple-900/30 rounded-2xl p-8 border border-pink-500/30 text-center">
          <Trophy size={64} className="mx-auto text-yellow-400 mb-4" />
          <h3 className="text-2xl font-bold text-white mb-2">Quiz Complete!</h3>
          <p className="text-4xl font-bold text-pink-400 mb-4">
            {score} / {questions.length}
          </p>
          <p className="text-gray-400 mb-6">
            {score === questions.length 
              ? 'Perfect score! You\'ve mastered Word2Vec!' 
              : score >= questions.length * 0.7
              ? 'Great job! You have a solid understanding.'
              : 'Keep practicing! Review the concepts and try again.'}
          </p>
          <button
            onClick={resetQuiz}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-pink-600 hover:bg-pink-700 rounded-lg transition-colors"
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
              <span className="px-3 py-1 bg-pink-600 rounded-full text-sm">
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
                className="flex items-center gap-2 px-6 py-3 bg-pink-600 hover:bg-pink-700 rounded-lg transition-colors"
              >
                Next Question
                <ArrowRight size={18} />
              </button>
            </div>
          )}
        </>
      )}

      {/* Quick Reference */}
      <div className="bg-gradient-to-r from-pink-900/20 to-purple-900/20 rounded-xl p-6 border border-pink-500/30">
        <h4 className="flex items-center gap-2 font-bold text-pink-400 mb-4">
          <HelpCircle size={18} />
          Quick Reference
        </h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Skip-gram</p>
            <p className="text-gray-400">Center word → Context words</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">CBOW</p>
            <p className="text-gray-400">Context words → Center word</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-orange-400 font-medium mb-1">Negative Sampling</p>
            <p className="text-gray-400">Binary classification instead of softmax</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-purple-400 font-medium mb-1">Embedding Dimension</p>
            <p className="text-gray-400">Typically 100-300</p>
          </div>
        </div>
      </div>
    </div>
  );
}
