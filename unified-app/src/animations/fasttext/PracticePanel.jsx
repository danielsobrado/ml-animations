import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Dumbbell, Check, X, RefreshCw, Trophy, Brain } from 'lucide-react';

function PracticePanel() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [completed, setCompleted] = useState(false);

  const questions = [
    {
      question: 'What makes FastText different from Word2Vec?',
      options: [
        'Uses character n-grams for subword information',
        'Uses attention mechanisms',
        'Only works with English',
        'Requires labeled data'
      ],
      correct: 0,
      explanation: 'FastText represents words as bags of character n-grams, capturing morphological information that Word2Vec misses.'
    },
    {
      question: 'How does FastText handle out-of-vocabulary (OOV) words?',
      options: [
        'Returns a zero vector',
        'Throws an error',
        'Sums the vectors of its character n-grams',
        'Uses the nearest known word'
      ],
      correct: 2,
      explanation: 'FastText computes OOV word vectors by summing the vectors of their constituent character n-grams.'
    },
    {
      question: 'What is the default n-gram range in FastText?',
      options: [
        '1-2 characters',
        '2-4 characters',
        '3-6 characters',
        '5-10 characters'
      ],
      correct: 2,
      explanation: 'FastText uses character n-grams from 3 to 6 characters by default (minn=3, maxn=6).'
    },
    {
      question: 'For which type of language is FastText most beneficial?',
      options: [
        'Languages with simple morphology (e.g., Chinese)',
        'Morphologically rich languages (e.g., German, Turkish)',
        'Programming languages only',
        'All languages equally'
      ],
      correct: 1,
      explanation: 'FastText excels with morphologically rich languages because it captures word structure through n-grams (e.g., German compound words).'
    },
    {
      question: 'What boundary markers does FastText add to words?',
      options: [
        '[START] and [END]',
        '@ and @',
        '< and >',
        'No markers are added'
      ],
      correct: 2,
      explanation: 'FastText adds < at the beginning and > at the end of words before extracting n-grams, helping distinguish prefixes and suffixes.'
    },
    {
      question: 'Which training objective does FastText use?',
      options: [
        'Only CBOW',
        'Only Skip-gram',
        'GloVe-style co-occurrence',
        'Skip-gram or CBOW with subword information'
      ],
      correct: 3,
      explanation: 'FastText extends both Skip-gram and CBOW architectures by adding subword (character n-gram) information to the training process.'
    },
    {
      question: 'What is a key advantage of FastText for text classification?',
      options: [
        'It requires no training data',
        'It has a built-in fast text classifier',
        'It only works with pre-defined categories',
        'It uses transformer architecture'
      ],
      correct: 1,
      explanation: 'FastText includes an efficient text classifier that can be trained with labeled data using train_supervised().'
    },
    {
      question: 'How does FastText compute a word vector?',
      options: [
        'Only uses the word itself',
        'Only uses context words',
        'Sums the word vector and all n-gram vectors',
        'Averages transformer attention'
      ],
      correct: 2,
      explanation: 'The final word vector is the sum of the vector for the word itself plus all its character n-gram vectors.'
    },
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
      setCompleted(true);
    }
  };

  const resetQuiz = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setCompleted(false);
  };

  if (completed) {
    const percentage = (score / questions.length) * 100;
    return (
      <div className="space-y-6">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-center py-12"
        >
          <Trophy className={`w-20 h-20 mx-auto mb-6 ${percentage >= 70 ? 'text-yellow-400' : 'text-purple-400'}`} />
          <h2 className="text-3xl font-bold text-white mb-2">Quiz Complete!</h2>
          <p className="text-purple-200/80 text-xl mb-6">
            You scored {score} out of {questions.length} ({percentage.toFixed(0)}%)
          </p>
          
          <div className="max-w-md mx-auto mb-8">
            <div className="h-4 bg-white/10 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${percentage}%` }}
                transition={{ duration: 1, delay: 0.5 }}
                className={`h-full ${percentage >= 70 ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-gradient-to-r from-purple-500 to-pink-500'}`}
              />
            </div>
          </div>

          <div className="text-purple-300/80 mb-8">
            {percentage >= 90 && "🎉 Excellent! You're a FastText expert!"}
            {percentage >= 70 && percentage < 90 && "👏 Great job! You have a solid understanding."}
            {percentage >= 50 && percentage < 70 && "📚 Good effort! Review the concepts and try again."}
            {percentage < 50 && "💪 Keep learning! Review the panels and try again."}
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={resetQuiz}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl font-medium text-white"
          >
            <RefreshCw className="w-5 h-5" />
            Try Again
          </motion.button>
        </motion.div>
      </div>
    );
  }

  const question = questions[currentQuestion];

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Practice Quiz</h2>
        <p className="text-purple-200/70">
          Test your understanding of FastText
        </p>
      </div>

      {/* Progress */}
      <div className="flex items-center justify-between mb-6">
        <span className="text-purple-300">
          Question {currentQuestion + 1} of {questions.length}
        </span>
        <span className="text-purple-300">
          Score: {score}/{currentQuestion + (showResult ? 1 : 0)}
        </span>
      </div>

      <div className="h-2 bg-white/10 rounded-full overflow-hidden mb-6">
        <motion.div
          animate={{ width: `${((currentQuestion + (showResult ? 1 : 0)) / questions.length) * 100}%` }}
          className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
        />
      </div>

      {/* Question */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentQuestion}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          className="bg-white/5 rounded-xl p-6"
        >
          <div className="flex items-start gap-3 mb-6">
            <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400 mt-1 flex-shrink-0" />
            <h3 className="text-xl font-medium text-white">{question.question}</h3>
          </div>

          <div className="space-y-3">
            {question.options.map((option, index) => (
              <motion.button
                key={index}
                whileHover={!showResult ? { scale: 1.02 } : {}}
                whileTap={!showResult ? { scale: 0.98 } : {}}
                onClick={() => !showResult && handleAnswer(index)}
                disabled={showResult}
                className={`w-full p-4 rounded-xl text-left transition-all flex items-center gap-3 ${
                  showResult
                    ? index === question.correct
                      ? 'bg-green-500/20 border-2 border-green-400'
                      : selectedAnswer === index
                      ? 'bg-red-500/20 border-2 border-red-400'
                      : 'bg-white/5 border-2 border-transparent'
                    : 'bg-white/10 hover:bg-white/20 border-2 border-transparent'
                }`}
              >
                <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  showResult && index === question.correct
                    ? 'bg-green-500 text-white'
                    : showResult && selectedAnswer === index
                    ? 'bg-red-500 text-white'
                    : 'bg-white/10 text-purple-300'
                }`}>
                  {showResult && index === question.correct ? (
                    <Check className="w-5 h-5" />
                  ) : showResult && selectedAnswer === index ? (
                    <X className="w-5 h-5" />
                  ) : (
                    String.fromCharCode(65 + index)
                  )}
                </span>
                <span className={showResult && index === question.correct ? 'text-green-300' : 'text-purple-100'}>
                  {option}
                </span>
              </motion.button>
            ))}
          </div>

          {/* Explanation */}
          <AnimatePresence>
            {showResult && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-6 p-4 bg-purple-500/20 rounded-xl border border-purple-400/30"
              >
                <p className="text-purple-200">
                  <strong className="text-purple-300">Explanation:</strong> {question.explanation}
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </AnimatePresence>

      {/* Next Button */}
      {showResult && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={nextQuestion}
            className="px-8 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl font-medium text-white"
          >
            {currentQuestion < questions.length - 1 ? 'Next Question' : 'See Results'}
          </motion.button>
        </motion.div>
      )}
    </div>
  );
}

export default PracticePanel;
