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
      question: 'What does BoW (Bag of Words) primarily capture?',
      options: [
        'Word order and grammar',
        'Word frequencies in a document',
        'Semantic meaning of words',
        'Syntactic relationships'
      ],
      correct: 1,
      explanation: 'BoW captures word frequencies by counting how many times each word appears in a document, completely ignoring word order.'
    },
    {
      type: 'multiple',
      question: 'In TF-IDF, what happens to words that appear in ALL documents?',
      options: [
        'They get the highest weight',
        'They are removed from vocabulary',
        'They get weight close to zero',
        'They are counted twice'
      ],
      correct: 2,
      explanation: 'Words appearing in all documents have DF = N, so IDF = log(N/N) = log(1) = 0. TF × 0 = 0, giving them zero weight.'
    },
    {
      type: 'multiple',
      question: 'What is the IDF formula?',
      options: [
        'log(DF / N)',
        'log(N × DF)',
        'log(N / DF)',
        'N / log(DF)'
      ],
      correct: 2,
      explanation: 'IDF = log(N / DF) where N is total documents and DF is the document frequency (how many docs contain the term).'
    },
    {
      type: 'calculation',
      question: 'Document has 10 words. "cat" appears 2 times. What is TF("cat")?',
      options: ['0.1', '0.2', '2.0', '0.5'],
      correct: 1,
      explanation: 'TF = count / total_words = 2 / 10 = 0.2'
    },
    {
      type: 'calculation',
      question: 'Corpus has 100 documents. "algorithm" appears in 10 documents. What is IDF("algorithm")? (Use natural log)',
      options: ['2.3', '1.0', '10.0', '0.1'],
      correct: 0,
      explanation: 'IDF = log(N / DF) = log(100 / 10) = log(10) ≈ 2.303'
    },
    {
      type: 'multiple',
      question: 'Which limitation is shared by BOTH BoW and TF-IDF?',
      options: [
        'Cannot handle rare words',
        'Loses word order information',
        'Only works with English',
        'Cannot handle large vocabularies'
      ],
      correct: 1,
      explanation: 'Both BoW and TF-IDF treat documents as unordered collections of words. "Dog bites man" and "Man bites dog" would have identical representations.'
    },
    {
      type: 'multiple',
      question: 'Why might TF-IDF be better than raw BoW for search engines?',
      options: [
        'It runs faster',
        'It uses less memory',
        'It down-weights common words automatically',
        'It understands word meanings'
      ],
      correct: 2,
      explanation: 'TF-IDF automatically reduces the importance of common words (like "the", "is", "a") that don\'t help distinguish documents, making search results more relevant.'
    },
    {
      type: 'multiple',
      question: 'Given vocabulary [and, cat, dog, the], what is the BoW vector for "the cat and the cat"?',
      options: [
        '[1, 2, 0, 2]',
        '[1, 1, 0, 2]',
        '[1, 2, 1, 2]',
        '[2, 1, 0, 2]'
      ],
      correct: 0,
      explanation: 'Count each word: "and"=1, "cat"=2, "dog"=0, "the"=2 → [1, 2, 0, 2]'
    },
    {
      type: 'multiple',
      question: 'What sklearn class would you use for TF-IDF vectorization?',
      options: [
        'CountVectorizer',
        'TfidfVectorizer',
        'HashingVectorizer',
        'OneHotEncoder'
      ],
      correct: 1,
      explanation: 'TfidfVectorizer from sklearn.feature_extraction.text computes TF-IDF scores. CountVectorizer gives raw counts (BoW).'
    },
    {
      type: 'multiple',
      question: 'If TF = 0.5 and IDF = 2.0, what is the TF-IDF score?',
      options: [
        '0.25',
        '1.0',
        '2.5',
        '4.0'
      ],
      correct: 1,
      explanation: 'TF-IDF = TF × IDF = 0.5 × 2.0 = 1.0'
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
          <span className="text-cyan-400">Practice</span> Quiz
        </h2>
        <p className="text-gray-400">
          Test your understanding of Bag of Words and TF-IDF
        </p>
      </div>

      {/* Progress */}
      <div className="bg-black/30 rounded-xl p-4 border border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400">Progress</span>
          <span className="text-sm text-cyan-400">{answered.length} / {questions.length}</span>
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
                  ? 'bg-cyan-500'
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
        <div className="bg-gradient-to-r from-cyan-900/30 to-purple-900/30 rounded-2xl p-8 border border-cyan-500/30 text-center">
          <Trophy size={64} className="mx-auto text-yellow-400 mb-4" />
          <h3 className="text-2xl font-bold text-white mb-2">Quiz Complete!</h3>
          <p className="text-4xl font-bold text-cyan-400 mb-4">
            {score} / {questions.length}
          </p>
          <p className="text-gray-400 mb-6">
            {score === questions.length 
              ? 'Perfect score! You\'ve mastered BoW and TF-IDF!' 
              : score >= questions.length * 0.7
              ? 'Great job! You have a solid understanding.'
              : 'Keep practicing! Review the concepts and try again.'}
          </p>
          <button
            onClick={resetQuiz}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors"
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
              <span className="px-3 py-1 bg-cyan-600 rounded-full text-sm">
                Question {currentQuestion + 1}
              </span>
              {questions[currentQuestion].type === 'calculation' && (
                <span className="px-3 py-1 bg-yellow-600 rounded-full text-sm">
                  Calculation
                </span>
              )}
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
                className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors"
              >
                Next Question
                <ArrowRight size={18} />
              </button>
            </div>
          )}
        </>
      )}

      {/* Quick Reference */}
      <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 rounded-xl p-6 border border-cyan-500/30">
        <h4 className="flex items-center gap-2 font-bold text-cyan-400 mb-4">
          <HelpCircle size={18} />
          Quick Reference
        </h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Term Frequency (TF)</p>
            <p className="text-gray-400">count(word, doc) / |doc|</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-yellow-400 font-medium mb-1">Inverse Document Frequency (IDF)</p>
            <p className="text-gray-400">log(N / DF(word))</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-purple-400 font-medium mb-1">TF-IDF</p>
            <p className="text-gray-400">TF × IDF</p>
          </div>
          <div className="bg-black/30 rounded-lg p-3">
            <p className="text-blue-400 font-medium mb-1">Key Insight</p>
            <p className="text-gray-400">High TF-IDF = common in doc, rare in corpus</p>
          </div>
        </div>
      </div>
    </div>
  );
}
