import React, { useState } from 'react';
import { FlaskConical, CheckCircle, XCircle, RotateCcw, Code, BookOpen, Lightbulb } from 'lucide-react';

function PracticePanel() {
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [activeExercise, setActiveExercise] = useState(0);

  const questions = [
    {
      id: 1,
      question: "What tokenizer does CLIP use?",
      options: ["WordPiece", "SentencePiece", "Byte Pair Encoding (BPE)", "Character-level"],
      correct: 2,
      explanation: "CLIP uses BPE (Byte Pair Encoding) with a vocabulary of ~49,000 tokens."
    },
    {
      id: 2,
      question: "What is the maximum token length for CLIP's text encoder?",
      options: ["64 tokens", "77 tokens", "128 tokens", "256 tokens"],
      correct: 1,
      explanation: "CLIP is limited to 77 tokens including [BOS] and [EOS] special tokens."
    },
    {
      id: 3,
      question: "What type of attention does CLIP's text encoder use?",
      options: ["Bidirectional", "Cross-attention", "Causal (unidirectional)", "Sparse attention"],
      correct: 2,
      explanation: "CLIP uses causal attention - each token can only attend to itself and previous tokens."
    },
    {
      id: 4,
      question: "What is the hidden dimension of CLIP-L/14?",
      options: ["512", "768", "1024", "1280"],
      correct: 1,
      explanation: "CLIP-L (Large) has a hidden dimension of 768. CLIP-G (Giant) has 1280."
    },
    {
      id: 5,
      question: "How is the 'pooled' embedding extracted from CLIP?",
      options: [
        "Average of all token embeddings",
        "First token [BOS] hidden state",
        "Last token [EOS] hidden state",
        "Max pooling across all tokens"
      ],
      correct: 2,
      explanation: "The pooled embedding is the hidden state at the [EOS] position, which sees all previous tokens due to causal attention."
    },
    {
      id: 6,
      question: "Why does SD3 use TWO CLIP models (CLIP-L and CLIP-G)?",
      options: [
        "Faster inference",
        "Redundancy for error correction",
        "Complementary representations at different scales",
        "One for text, one for images"
      ],
      correct: 2,
      explanation: "CLIP-L and CLIP-G provide complementary representations - different model sizes capture different aspects of the text."
    },
    {
      id: 7,
      question: "What is the main advantage of T5 over CLIP for text encoding?",
      options: [
        "Faster inference speed",
        "Visual alignment",
        "Bidirectional attention and longer context",
        "Smaller model size"
      ],
      correct: 2,
      explanation: "T5 uses bidirectional attention (better understanding) and supports much longer sequences (256+ tokens vs 77)."
    },
    {
      id: 8,
      question: "Where are CLIP's sequence embeddings used in SD3?",
      options: [
        "Only for VAE encoding",
        "Joint attention with image tokens",
        "Only for final image decoding",
        "Not used in SD3"
      ],
      correct: 1,
      explanation: "CLIP's sequence embeddings are concatenated with image tokens for joint attention in the DiT transformer."
    },
  ];

  const exercises = [
    {
      title: "Exercise 1: Calculate Token Embedding Shape",
      description: "Given a batch of 4 text prompts encoded with CLIP-L, what is the shape of the output embeddings?",
      task: `Prompt batch size: 4
Max tokens: 77
CLIP-L hidden dimension: 768

Calculate:
1. Token embedding shape (before transformer)
2. Sequence output shape (after transformer)
3. Pooled embedding shape`,
      solution: `1. Token embedding shape: [4, 77, 768]
   - 4 prompts × 77 tokens × 768 dimensions

2. Sequence output shape: [4, 77, 768]
   - Same shape, but now contextualized

3. Pooled embedding shape: [4, 768]
   - One 768-d vector per prompt (from [EOS])`
    },
    {
      title: "Exercise 2: Combined CLIP Embeddings",
      description: "SD3 uses both CLIP-L (768-d) and CLIP-G (1280-d). Calculate the combined dimensions.",
      task: `For joint attention in SD3:
- CLIP-L sequence: [B, 77, 768]
- CLIP-G sequence: [B, 77, 1280]

For DiT conditioning:
- CLIP-L pooled: [B, 768]
- CLIP-G pooled: [B, 1280]

How are these combined?`,
      solution: `For joint attention:
- Concatenate on feature dimension
- Combined: [B, 77, 768 + 1280] = [B, 77, 2048]

For DiT conditioning:
- Concatenate pooled embeddings
- Combined: [B, 768 + 1280] = [B, 2048]
- Then projected to match DiT hidden size`
    },
    {
      title: "Exercise 3: Implement CLIP Tokenization",
      description: "Write pseudocode for CLIP's text preprocessing pipeline.",
      task: `Implement a function that:
1. Takes raw text input
2. Tokenizes using BPE
3. Adds special tokens
4. Pads to 77 tokens
5. Returns token IDs`,
      solution: `def clip_tokenize(text, max_length=77):
    # Step 1: Clean and lowercase
    text = text.lower().strip()
    
    # Step 2: BPE tokenization
    tokens = bpe_tokenizer.encode(text)
    
    # Step 3: Add special tokens
    bos_token = 49406  # [BOS]
    eos_token = 49407  # [EOS]
    pad_token = 0      # [PAD]
    
    # Step 4: Truncate if needed
    max_text_tokens = max_length - 2  # Reserve for BOS/EOS
    tokens = tokens[:max_text_tokens]
    
    # Step 5: Construct sequence
    token_ids = [bos_token] + tokens + [eos_token]
    
    # Step 6: Pad to max_length
    padding_length = max_length - len(token_ids)
    token_ids = token_ids + [pad_token] * padding_length
    
    return token_ids  # Shape: [77]`
    },
    {
      title: "Exercise 4: Causal vs Bidirectional Attention",
      description: "Explain the attention mask difference between CLIP and T5.",
      task: `For the sequence: [BOS, A, cat, sat, EOS]

Draw the attention mask for:
1. CLIP (causal attention)
2. T5 (bidirectional attention)

Mark with 1 where attention is allowed, 0 where blocked.`,
      solution: `CLIP Causal Attention Mask:
       BOS  A  cat sat EOS
BOS  [  1   0   0   0   0  ]
A    [  1   1   0   0   0  ]
cat  [  1   1   1   0   0  ]
sat  [  1   1   1   1   0  ]
EOS  [  1   1   1   1   1  ]

T5 Bidirectional Attention Mask:
       BOS  A  cat sat EOS
BOS  [  1   1   1   1   1  ]
A    [  1   1   1   1   1  ]
cat  [  1   1   1   1   1  ]
sat  [  1   1   1   1   1  ]
EOS  [  1   1   1   1   1  ]

Key insight: In CLIP, only EOS sees all tokens!`
    }
  ];

  const handleAnswer = (questionId, optionIndex) => {
    setAnswers({ ...answers, [questionId]: optionIndex });
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach(q => {
      if (answers[q.id] === q.correct) correct++;
    });
    return correct;
  };

  const resetQuiz = () => {
    setAnswers({});
    setShowResults(false);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-blue-400 mb-2">Practice Lab</h2>
        <p className="text-gray-300">
          Test your understanding of CLIP text encoders with quizzes and exercises.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setActiveExercise(-1)}
          className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
            activeExercise === -1
              ? 'bg-blue-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
        >
          <BookOpen size={18} />
          Quiz
        </button>
        <button
          onClick={() => setActiveExercise(0)}
          className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
            activeExercise >= 0
              ? 'bg-purple-600 text-white'
              : 'bg-white/10 text-gray-300 hover:bg-white/20'
          }`}
        >
          <Code size={18} />
          Exercises
        </button>
      </div>

      {activeExercise === -1 ? (
        /* Quiz Section */
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-300">
              CLIP Text Encoder Quiz
            </h3>
            <button
              onClick={resetQuiz}
              className="flex items-center gap-2 px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm transition-colors"
            >
              <RotateCcw size={16} />
              Reset
            </button>
          </div>

          {questions.map((q, idx) => (
            <div key={q.id} className="bg-black/30 rounded-xl p-6 border border-white/10">
              <div className="flex items-start gap-3 mb-4">
                <span className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold shrink-0">
                  {idx + 1}
                </span>
                <p className="text-gray-200 font-medium">{q.question}</p>
              </div>

              <div className="space-y-2 ml-11">
                {q.options.map((option, optIdx) => {
                  const isSelected = answers[q.id] === optIdx;
                  const isCorrect = optIdx === q.correct;
                  const showCorrect = showResults && isCorrect;
                  const showWrong = showResults && isSelected && !isCorrect;

                  return (
                    <button
                      key={optIdx}
                      onClick={() => !showResults && handleAnswer(q.id, optIdx)}
                      disabled={showResults}
                      className={`w-full text-left px-4 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                        showCorrect
                          ? 'bg-green-500/20 border-green-500'
                          : showWrong
                          ? 'bg-red-500/20 border-red-500'
                          : isSelected
                          ? 'bg-blue-500/30 border-blue-500'
                          : 'bg-white/5 border-transparent hover:bg-white/10'
                      } border`}
                    >
                      <span className={`w-6 h-6 rounded-full border flex items-center justify-center text-sm ${
                        isSelected ? 'border-blue-400 text-blue-400' : 'border-gray-500 text-gray-500'
                      }`}>
                        {String.fromCharCode(65 + optIdx)}
                      </span>
                      <span className="text-gray-300">{option}</span>
                      {showCorrect && <CheckCircle className="ml-auto text-green-400" size={20} />}
                      {showWrong && <XCircle className="ml-auto text-red-400" size={20} />}
                    </button>
                  );
                })}
              </div>

              {showResults && (
                <div className={`mt-4 ml-11 p-3 rounded-lg ${
                  answers[q.id] === q.correct ? 'bg-green-500/10' : 'bg-yellow-500/10'
                }`}>
                  <div className="flex items-center gap-2 text-sm">
                    <Lightbulb size={16} className="text-yellow-400" />
                    <span className="text-gray-300">{q.explanation}</span>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Submit/Results */}
          <div className="text-center">
            {!showResults ? (
              <button
                onClick={() => setShowResults(true)}
                disabled={Object.keys(answers).length < questions.length}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 rounded-lg font-semibold transition-colors"
              >
                Submit Answers
              </button>
            ) : (
              <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl p-6 border border-purple-500/30">
                <div className="text-3xl font-bold text-white mb-2">
                  {calculateScore()} / {questions.length}
                </div>
                <p className="text-gray-300">
                  {calculateScore() === questions.length
                    ? "🎉 Perfect! You've mastered CLIP text encoders!"
                    : calculateScore() >= questions.length * 0.7
                    ? "👍 Great job! Review the explanations for any missed questions."
                    : "📚 Keep learning! Review the previous panels and try again."}
                </p>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Exercises Section */
        <div className="space-y-6">
          {/* Exercise Navigation */}
          <div className="flex flex-wrap gap-2">
            {exercises.map((ex, idx) => (
              <button
                key={idx}
                onClick={() => setActiveExercise(idx)}
                className={`px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeExercise === idx
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
              >
                Exercise {idx + 1}
              </button>
            ))}
          </div>

          {/* Active Exercise */}
          <div className="bg-black/30 rounded-xl p-6 border border-purple-500/30">
            <h3 className="text-xl font-semibold text-purple-400 mb-2">
              {exercises[activeExercise].title}
            </h3>
            <p className="text-gray-300 mb-4">
              {exercises[activeExercise].description}
            </p>

            <div className="bg-black/40 rounded-lg p-4 mb-4">
              <div className="text-sm text-gray-400 mb-2">Task:</div>
              <pre className="text-gray-200 whitespace-pre-wrap font-mono text-sm">
                {exercises[activeExercise].task}
              </pre>
            </div>

            <details className="group">
              <summary className="cursor-pointer px-4 py-2 bg-purple-600/30 hover:bg-purple-600/40 rounded-lg text-purple-300 transition-colors">
                Show Solution
              </summary>
              <div className="mt-4 bg-green-500/10 rounded-lg p-4 border border-green-500/30">
                <pre className="text-green-300 whitespace-pre-wrap font-mono text-sm">
                  {exercises[activeExercise].solution}
                </pre>
              </div>
            </details>
          </div>

          {/* Tips */}
          <div className="bg-yellow-500/10 rounded-xl p-4 border border-yellow-500/30">
            <h4 className="text-yellow-400 font-semibold mb-2 flex items-center gap-2">
              <Lightbulb size={18} />
              Tips for Success
            </h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Remember: CLIP-L = 768-d, CLIP-G = 1280-d</li>
              <li>• Max tokens is always 77 (including BOS and EOS)</li>
              <li>• Causal attention = lower triangular mask</li>
              <li>• Pooled embedding comes from [EOS] token position</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default PracticePanel;
