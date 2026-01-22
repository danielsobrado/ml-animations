import React, { useState } from 'react';
import { FlaskConical, CheckCircle, XCircle, RotateCcw, Code, BookOpen, Lightbulb } from 'lucide-react';

function PracticePanel() {
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [activeExercise, setActiveExercise] = useState(0);

  const questions = [
    {
      id: 1,
      question: "What tokenizer does T5 use?",
      options: ["BPE", "WordPiece", "SentencePiece", "Character-level"],
      correct: 2,
      explanation: "T5 uses SentencePiece with a vocabulary of ~32,000 tokens and the ▁ marker for word boundaries."
    },
    {
      id: 2,
      question: "How many encoder layers does T5-XXL have?",
      options: ["12", "24", "32", "48"],
      correct: 1,
      explanation: "T5-XXL has 24 encoder layers (and 24 decoder layers, but SD3 only uses the encoder)."
    },
    {
      id: 3,
      question: "What is the hidden dimension of T5-XXL?",
      options: ["768", "1024", "2048", "4096"],
      correct: 3,
      explanation: "T5-XXL has a hidden dimension of 4096, much larger than CLIP's 768/1280."
    },
    {
      id: 4,
      question: "What type of attention does T5's encoder use?",
      options: ["Causal (unidirectional)", "Bidirectional", "Cross-attention only", "Sparse attention"],
      correct: 1,
      explanation: "T5's encoder uses bidirectional self-attention - every token can attend to all other tokens."
    },
    {
      id: 5,
      question: "What activation function does T5 use in its feed-forward layers?",
      options: ["ReLU", "GELU", "GEGLU", "SiLU"],
      correct: 2,
      explanation: "T5 uses GEGLU (Gaussian Error Gated Linear Unit) which combines GELU with a gating mechanism."
    },
    {
      id: 6,
      question: "Why does SD3 only use T5's encoder (not decoder)?",
      options: [
        "The decoder is broken",
        "Encoders are faster",
        "SD3 needs text understanding, not text generation",
        "The decoder uses too much memory"
      ],
      correct: 2,
      explanation: "SD3 needs rich text embeddings to condition image generation. The encoder provides understanding; the decoder would generate text, which isn't needed."
    },
    {
      id: 7,
      question: "What is T5's advantage over CLIP for long prompts?",
      options: [
        "Faster processing",
        "Bidirectional attention + longer context",
        "Visual alignment",
        "Smaller model size"
      ],
      correct: 1,
      explanation: "T5 supports 256+ tokens (vs CLIP's 77) and uses bidirectional attention for better understanding of complex relationships."
    },
    {
      id: 8,
      question: "Approximately how much VRAM does T5-XXL require?",
      options: ["~2GB", "~4GB", "~8-10GB", "~20GB"],
      correct: 2,
      explanation: "T5-XXL's encoder requires approximately 8-10GB VRAM, making it optional for users with limited resources."
    },
  ];

  const exercises = [
    {
      title: "Exercise 1: T5 Output Dimensions",
      description: "Calculate the output shapes for different T5 variants.",
      task: `Given a batch of 2 prompts with 50 tokens each, calculate the encoder output shape for:

1. T5-Base (hidden_dim=768)
2. T5-Large (hidden_dim=1024)
3. T5-XXL (hidden_dim=4096)

Note: T5 doesn't pad to fixed length like CLIP.`,
      solution: `All T5 variants output shape: [batch_size, seq_len, hidden_dim]

1. T5-Base: [2, 50, 768]
   Total elements: 2 × 50 × 768 = 76,800

2. T5-Large: [2, 50, 1024]
   Total elements: 2 × 50 × 1024 = 102,400

3. T5-XXL: [2, 50, 4096]
   Total elements: 2 × 50 × 4096 = 409,600

T5-XXL has 5.3× more output elements than T5-Base!`
    },
    {
      title: "Exercise 2: Attention Comparison",
      description: "Compare attention patterns between CLIP and T5 for a sample sentence.",
      task: `For the sentence: "The cat sat on the mat"
(6 tokens, ignoring special tokens)

1. How many attention connections per token in CLIP (causal)?
2. How many attention connections per token in T5 (bidirectional)?
3. What's the total attention computations for each?`,
      solution: `CLIP (Causal Attention):
Token 1: sees 1 token
Token 2: sees 2 tokens
Token 3: sees 3 tokens
Token 4: sees 4 tokens
Token 5: sees 5 tokens
Token 6: sees 6 tokens
Total: 1+2+3+4+5+6 = 21 attention connections

T5 (Bidirectional Attention):
Each token sees all 6 tokens
Total: 6 × 6 = 36 attention connections

T5 has 36/21 = 1.7× more attention connections!

For n tokens:
- CLIP: n(n+1)/2 connections
- T5: n² connections`
    },
    {
      title: "Exercise 3: SentencePiece Tokenization",
      description: "Practice SentencePiece tokenization patterns.",
      task: `Predict how SentencePiece would tokenize these phrases:

1. "hello world"
2. "unbelievable"
3. "artificial intelligence"
4. "SD3 is amazing"

Remember: ▁ marks word boundaries (including leading space)`,
      solution: `1. "hello world"
   → ["▁hello", "▁world"]
   (Each word gets ▁ prefix for space)

2. "unbelievable"
   → ["▁un", "believ", "able"]
   (Common prefix + suffix split)

3. "artificial intelligence"  
   → ["▁art", "ificial", "▁intellig", "ence"]
   (Long words split into subwords)

4. "SD3 is amazing"
   → ["▁SD", "3", "▁is", "▁amazing"]
   (Numbers often separate, common words intact)

Key insight: ▁ appears at START of words (not end), 
representing the space BEFORE the word.`
    },
    {
      title: "Exercise 4: Memory Calculation",
      description: "Estimate memory requirements for T5 encoding.",
      task: `Calculate the memory needed to store T5-XXL encoder outputs for a batch:

Batch size: 8
Sequence length: 256 tokens
Hidden dimension: 4096
Data type: float16 (2 bytes per value)

Also calculate for float32 (4 bytes per value).`,
      solution: `Output tensor shape: [8, 256, 4096]
Total elements: 8 × 256 × 4096 = 8,388,608 elements

float16 memory:
8,388,608 × 2 bytes = 16,777,216 bytes
= 16 MB per batch

float32 memory:
8,388,608 × 4 bytes = 33,554,432 bytes
= 32 MB per batch

Note: This is just the OUTPUT. The model weights 
themselves require ~8-10GB for T5-XXL encoder!

Total during inference ≈ 10GB + batch outputs`
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
        <h2 className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">Practice Lab</h2>
        <p className="text-gray-700 dark:text-gray-300">
          Test your understanding of T5 text encoders with quizzes and exercises.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={() => setActiveExercise(-1)}
          className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
            activeExercise === -1
              ? 'bg-emerald-600 text-white'
              : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
          }`}
        >
          <BookOpen size={18} />
          Quiz
        </button>
        <button
          onClick={() => setActiveExercise(0)}
          className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
            activeExercise >= 0
              ? 'bg-teal-600 text-white'
              : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
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
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">
              T5 Text Encoder Quiz
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
                <span className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-white font-bold shrink-0">
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
                          ? 'bg-emerald-500/30 border-emerald-500'
                          : 'bg-white/5 border-transparent hover:bg-white/10'
                      } border`}
                    >
                      <span className={`w-6 h-6 rounded-full border flex items-center justify-center text-sm ${
                        isSelected ? 'border-emerald-400 text-emerald-600 dark:text-emerald-400' : 'border-gray-500 text-gray-700 dark:text-gray-500'
                      }`}>
                        {String.fromCharCode(65 + optIdx)}
                      </span>
                      <span className="text-gray-700 dark:text-gray-300">{option}</span>
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
                    <span className="text-gray-700 dark:text-gray-300">{q.explanation}</span>
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
                className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50 rounded-lg font-semibold transition-colors"
              >
                Submit Answers
              </button>
            ) : (
              <div className="bg-gradient-to-r from-emerald-500/20 to-teal-500/20 rounded-xl p-6 border border-emerald-500/30">
                <div className="text-3xl font-bold text-white mb-2">
                  {calculateScore()} / {questions.length}
                </div>
                <p className="text-gray-700 dark:text-gray-300">
                  {calculateScore() === questions.length
                    ? "🎉 Perfect! You've mastered T5 text encoders!"
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
                    ? 'bg-teal-600 text-white'
                    : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
                }`}
              >
                Exercise {idx + 1}
              </button>
            ))}
          </div>

          {/* Active Exercise */}
          <div className="bg-black/30 rounded-xl p-6 border border-teal-500/30">
            <h3 className="text-xl font-semibold text-teal-600 dark:text-teal-400 mb-2">
              {exercises[activeExercise].title}
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              {exercises[activeExercise].description}
            </p>

            <div className="bg-black/40 rounded-lg p-4 mb-4">
              <div className="text-sm text-gray-800 dark:text-gray-400 mb-2">Task:</div>
              <pre className="text-gray-200 whitespace-pre-wrap font-mono text-sm">
                {exercises[activeExercise].task}
              </pre>
            </div>

            <details className="group">
              <summary className="cursor-pointer px-4 py-2 bg-teal-600/30 hover:bg-teal-600/40 rounded-lg text-teal-300 transition-colors">
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
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• T5-XXL hidden_dim = 4096 (much larger than CLIP)</li>
              <li>• Bidirectional attention = n² attention computations</li>
              <li>• SentencePiece uses ▁ to mark word boundaries</li>
              <li>• T5 encoder output shape: [batch, seq_len, hidden_dim]</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default PracticePanel;
