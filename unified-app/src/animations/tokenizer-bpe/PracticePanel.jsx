import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Code, Lightbulb, Trophy, ChevronRight } from 'lucide-react';

function PracticePanel() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [showExercise, setShowExercise] = useState(false);

  const questions = [
    {
      question: "What is Byte-Pair Encoding (BPE)?",
      options: [
        "A compression algorithm for images",
        "A subword tokenization method that merges frequent character pairs",
        "A way to encrypt text data",
        "A method to count word frequencies"
      ],
      correct: 1,
      explanation: "BPE iteratively merges the most frequent adjacent pairs of characters/tokens, building a vocabulary of subword units."
    },
    {
      question: "Why does CLIP's vocabulary contain the '</w>' marker?",
      options: [
        "To indicate errors in tokenization",
        "To mark word boundaries (end of word)",
        "To represent whitespace",
        "To indicate numbers"
      ],
      correct: 1,
      explanation: "The '</w>' suffix marks the end of a word, allowing reconstruction of original spacing and distinguishing 'notebook' from 'note book'."
    },
    {
      question: "What is CLIP's maximum token context length?",
      options: [
        "512 tokens",
        "256 tokens",
        "77 tokens",
        "4096 tokens"
      ],
      correct: 2,
      explanation: "CLIP has a fixed context of 77 tokens (75 text + BOS + EOS), which limits detailed prompts but works well for image-text alignment."
    },
    {
      question: "How does T5's tokenization differ from CLIP's?",
      options: [
        "T5 uses word-level only",
        "T5 uses SentencePiece with underscore prefix for word starts",
        "T5 doesn't use special tokens",
        "T5 has a smaller vocabulary"
      ],
      correct: 1,
      explanation: "T5 uses SentencePiece which marks word boundaries with a '▁' prefix at the START of words, unlike CLIP's '</w>' suffix at the END."
    },
    {
      question: "What happens when a token sequence exceeds max_length?",
      options: [
        "It causes an error",
        "Extra tokens are ignored",
        "The sequence is truncated and EOS is added at the end",
        "The vocabulary expands automatically"
      ],
      correct: 2,
      explanation: "Sequences are truncated to fit, with the end-of-sequence token placed at the final position to properly terminate the sequence."
    },
    {
      question: "Why does SD3 use both CLIP and T5 tokenizers?",
      options: [
        "For redundancy in case one fails",
        "CLIP has visual grounding, T5 has better language understanding",
        "To double the vocabulary size",
        "T5 is faster than CLIP"
      ],
      correct: 1,
      explanation: "CLIP provides visual concept understanding from image-text training, while T5 offers sophisticated NLP capabilities for complex prompts."
    },
    {
      question: "What is the purpose of the <|startoftext|> token?",
      options: [
        "To mark paragraph breaks",
        "To signal the beginning of a tokenized sequence",
        "To indicate the prompt is a question",
        "To separate multiple prompts"
      ],
      correct: 1,
      explanation: "The start-of-text (BOS) token marks sequence boundaries, helping the model understand where input begins."
    },
    {
      question: "How are unknown words handled in BPE tokenization?",
      options: [
        "They are replaced with [UNK]",
        "They cause tokenization to fail",
        "They are broken into known subword pieces",
        "They are removed from the text"
      ],
      correct: 2,
      explanation: "BPE can tokenize ANY word by falling back to individual characters. 'Supercalifragilistic' becomes multiple subword tokens that exist in the vocabulary."
    }
  ];

  const handleAnswer = (index) => {
    if (showResult) return;
    setSelectedAnswer(index);
    setShowResult(true);
    if (index === questions[currentQuestion].correct) {
      setScore(score + 1);
    }
  };

  const nextQuestion = () => {
    setSelectedAnswer(null);
    setShowResult(false);
    setCurrentQuestion(currentQuestion + 1);
  };

  const resetQuiz = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setShowExercise(false);
  };

  const isComplete = currentQuestion >= questions.length;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-2">Practice Lab</h2>
        <p className="text-gray-700 dark:text-gray-300">Test your understanding of diffusion model tokenization</p>
      </div>

      {/* Mode Toggle */}
      <div className="flex justify-center gap-4">
        <button
          onClick={() => setShowExercise(false)}
          className={`px-6 py-3 rounded-xl font-semibold transition-all ${
            !showExercise ? 'bg-orange-500 text-white' : 'bg-black/30 text-gray-800 dark:text-gray-400 hover:text-white'
          }`}
        >
          Quiz
        </button>
        <button
          onClick={() => setShowExercise(true)}
          className={`px-6 py-3 rounded-xl font-semibold transition-all ${
            showExercise ? 'bg-orange-500 text-white' : 'bg-black/30 text-gray-800 dark:text-gray-400 hover:text-white'
          }`}
        >
          Code Exercises
        </button>
      </div>

      {!showExercise ? (
        /* Quiz Mode */
        <div className="bg-black/40 rounded-xl p-6">
          {!isComplete ? (
            <>
              {/* Progress */}
              <div className="flex justify-between items-center mb-6">
                <span className="text-gray-800 dark:text-gray-400">Question {currentQuestion + 1} of {questions.length}</span>
                <span className="text-orange-600 dark:text-orange-400 font-semibold">Score: {score}/{questions.length}</span>
              </div>

              {/* Question */}
              <div className="mb-6">
                <h3 className="text-xl text-white mb-6">{questions[currentQuestion].question}</h3>
                <div className="space-y-3">
                  {questions[currentQuestion].options.map((option, i) => (
                    <button
                      key={i}
                      onClick={() => handleAnswer(i)}
                      disabled={showResult}
                      className={`w-full text-left p-4 rounded-lg transition-all ${
                        showResult
                          ? i === questions[currentQuestion].correct
                            ? 'bg-green-500/20 border border-green-500'
                            : i === selectedAnswer
                              ? 'bg-red-500/20 border border-red-500'
                              : 'bg-black/30 border border-transparent'
                          : 'bg-black/30 hover:bg-black/50 border border-transparent hover:border-orange-500/50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                          showResult && i === questions[currentQuestion].correct
                            ? 'bg-green-500 text-white'
                            : showResult && i === selectedAnswer
                              ? 'bg-red-500 text-white'
                              : 'bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}>
                          {String.fromCharCode(65 + i)}
                        </span>
                        <span className="text-gray-200">{option}</span>
                        {showResult && i === questions[currentQuestion].correct && (
                          <CheckCircle className="ml-auto text-green-400" size={20} />
                        )}
                        {showResult && i === selectedAnswer && i !== questions[currentQuestion].correct && (
                          <XCircle className="ml-auto text-red-400" size={20} />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Explanation */}
              {showResult && (
                <div className={`p-4 rounded-lg mb-6 ${
                  selectedAnswer === questions[currentQuestion].correct
                    ? 'bg-green-500/10 border border-green-500/30'
                    : 'bg-red-500/10 border border-red-500/30'
                }`}>
                  <div className="flex items-start gap-2">
                    <Lightbulb size={18} className={
                      selectedAnswer === questions[currentQuestion].correct
                        ? 'text-green-400'
                        : 'text-red-400'
                    } />
                    <p className="text-gray-700 dark:text-sm">{questions[currentQuestion].explanation}</p>
                  </div>
                </div>
              )}

              {/* Next Button */}
              {showResult && (
                <button
                  onClick={nextQuestion}
                  className="flex items-center gap-2 px-6 py-3 bg-orange-500 hover:bg-orange-600 text-white rounded-xl transition-colors"
                >
                  Next Question <ChevronRight size={18} />
                </button>
              )}
            </>
          ) : (
            /* Quiz Complete */
            <div className="text-center py-8">
              <Trophy size={64} className="mx-auto text-yellow-400 mb-4" />
              <h3 className="text-2xl font-bold text-white mb-2">Quiz Complete!</h3>
              <p className="text-4xl font-bold text-orange-600 dark:text-orange-400 mb-4">{score} / {questions.length}</p>
              <p className="text-gray-800 dark:text-gray-400 mb-6">
                {score === questions.length ? "Perfect score! You're a tokenization expert!" :
                 score >= questions.length * 0.7 ? "Great job! You understand tokenization well." :
                 "Keep learning! Review the concepts and try again."}
              </p>
              <button
                onClick={resetQuiz}
                className="flex items-center gap-2 px-6 py-3 bg-orange-500 hover:bg-orange-600 text-white rounded-xl transition-colors mx-auto"
              >
                <RotateCcw size={18} /> Try Again
              </button>
            </div>
          )}
        </div>
      ) : (
        /* Code Exercises */
        <div className="space-y-6">
          {/* Exercise 1 */}
          <div className="bg-black/40 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Code className="text-orange-600 dark:text-orange-400" size={20} />
              <h3 className="text-lg font-semibold text-white">Exercise 1: Simple Tokenizer</h3>
            </div>
            <p className="text-gray-800 dark:text-sm mb-4">
              Implement a basic character-level tokenizer with special tokens:
            </p>
            <div className="bg-black/60 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-700 dark:text-gray-300">
{`class SimpleTokenizer:
    def __init__(self):
        # Build character vocabulary a-z, space
        self.vocab = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
        self.vocab[' '] = 26
        self.vocab['<START>'] = 27
        self.vocab['<END>'] = 28
        self.vocab['<PAD>'] = 29
    
    def encode(self, text: str, max_length: int = 20) -> list:
        # TODO: Implement tokenization
        # 1. Convert to lowercase
        # 2. Add START token
        # 3. Convert each character to ID
        # 4. Add END token  
        # 5. Pad to max_length
        pass

# Test
tokenizer = SimpleTokenizer()
print(tokenizer.encode("hello"))
# Expected: [27, 7, 4, 11, 11, 14, 28, 29, 29, ...]`}
              </pre>
            </div>
          </div>

          {/* Exercise 2 */}
          <div className="bg-black/40 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Code className="text-orange-600 dark:text-orange-400" size={20} />
              <h3 className="text-lg font-semibold text-white">Exercise 2: BPE Merge</h3>
            </div>
            <p className="text-gray-800 dark:text-sm mb-4">
              Implement one step of BPE merging:
            </p>
            <div className="bg-black/60 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-700 dark:text-gray-300">
{`def bpe_merge(tokens: list, pair: tuple) -> list:
    """
    Merge all occurrences of pair in tokens.
    
    Example:
    tokens = ['l', 'o', 'w', '</w>', 'l', 'o', 'w', 'e', 'r', '</w>']
    pair = ('l', 'o')
    result = ['lo', 'w', '</w>', 'lo', 'w', 'e', 'r', '</w>']
    """
    # TODO: Implement
    # Scan through tokens, merge adjacent pairs
    result = []
    i = 0
    while i < len(tokens):
        # Your code here
        pass
    return result

# Test
tokens = ['l', 'o', 'w', '</w>', 'l', 'o', 'w', 'e', 'r', '</w>']
print(bpe_merge(tokens, ('l', 'o')))`}
              </pre>
            </div>
          </div>

          {/* Exercise 3 */}
          <div className="bg-black/40 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Code className="text-orange-600 dark:text-orange-400" size={20} />
              <h3 className="text-lg font-semibold text-white">Exercise 3: Count Pairs</h3>
            </div>
            <p className="text-gray-800 dark:text-sm mb-4">
              Count adjacent token pairs (needed for BPE training):
            </p>
            <div className="bg-black/60 rounded-lg p-4 font-mono text-sm overflow-x-auto">
              <pre className="text-gray-700 dark:text-gray-300">
{`from collections import Counter

def count_pairs(tokens: list) -> Counter:
    """
    Count frequency of adjacent pairs.
    
    Example:
    tokens = ['t', 'h', 'e', '</w>', 't', 'h', 'e', 'r', 'e', '</w>']
    result = Counter({('t', 'h'): 2, ('h', 'e'): 2, ('e', '</w>'): 1, ...})
    """
    # TODO: Implement
    pass

# Test
tokens = ['t', 'h', 'e', '</w>', 't', 'h', 'e', 'r', 'e', '</w>']
pairs = count_pairs(tokens)
print(pairs.most_common(3))`}
              </pre>
            </div>
          </div>

          {/* Solution Hints */}
          <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl p-6 border border-purple-500/30">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3 flex items-center gap-2">
              <Lightbulb size={18} />
              Solution Hints
            </h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>Exercise 1:</strong> Use list comprehension with <code>self.vocab.get(c, 29)</code> for unknown chars</li>
              <li>• <strong>Exercise 2:</strong> Check if <code>tokens[i:i+2] == list(pair)</code>, merge if true, else append single token</li>
              <li>• <strong>Exercise 3:</strong> Use <code>zip(tokens, tokens[1:])</code> to get adjacent pairs</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default PracticePanel;
