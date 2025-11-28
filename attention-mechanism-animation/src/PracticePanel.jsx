import React, { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, RotateCcw, Lightbulb, Calculator, Brain, Zap } from 'lucide-react';

const quizQuestions = [
    {
        id: 1,
        question: 'What does the Query (Q) represent in attention?',
        options: [
            'The information we want to retrieve',
            'What we are looking for / searching with',
            'The final output of attention',
            'The normalization factor'
        ],
        correct: 1,
        explanation: 'The Query represents what a position is "looking for" - it\'s the search term that gets compared against all Keys to determine relevance.'
    },
    {
        id: 2,
        question: 'Why do we scale by √d_k in scaled dot-product attention?',
        options: [
            'To make the computation faster',
            'To reduce memory usage',
            'To prevent softmax saturation from large dot products',
            'To normalize the output dimensions'
        ],
        correct: 2,
        explanation: 'When d_k is large, dot products can become very large, pushing softmax into regions with extremely small gradients. Scaling by √d_k keeps the variance around 1.'
    },
    {
        id: 3,
        question: 'What is the purpose of using multiple attention heads?',
        options: [
            'To make the model larger',
            'To learn different types of relationships in parallel',
            'To speed up training',
            'To reduce overfitting'
        ],
        correct: 1,
        explanation: 'Multiple heads allow the model to attend to information from different representation subspaces - one head might focus on syntax, another on semantics, etc.'
    },
    {
        id: 4,
        question: 'In Multi-Head Attention with h heads and d_model dimensions, what is d_k (dimension per head)?',
        options: [
            'd_model × h',
            'd_model + h',
            'd_model / h',
            'h / d_model'
        ],
        correct: 2,
        explanation: 'd_k = d_model / h. For example, with d_model=512 and h=8 heads, each head operates on 64-dimensional subspaces.'
    },
    {
        id: 5,
        question: 'After computing all attention heads, what operation combines them?',
        options: [
            'Addition',
            'Average pooling',
            'Concatenation followed by linear projection',
            'Element-wise multiplication'
        ],
        correct: 2,
        explanation: 'All head outputs are concatenated along the feature dimension, then projected through W_O to get back to d_model dimensions.'
    },
    {
        id: 6,
        question: 'What makes attention "soft" compared to traditional database lookups?',
        options: [
            'It uses floating point numbers',
            'It returns a weighted combination of ALL values',
            'It\'s slower to compute',
            'It uses gradients'
        ],
        correct: 1,
        explanation: 'Unlike hard lookups that return one exact match, attention computes a weighted average of all values, allowing smooth, differentiable retrieval.'
    },
    {
        id: 7,
        question: 'In self-attention, where do Q, K, and V come from?',
        options: [
            'Q from encoder, K and V from decoder',
            'All three from the same input sequence',
            'Q and K from input, V from output',
            'They are random vectors'
        ],
        correct: 1,
        explanation: 'In self-attention, Q, K, and V are all derived from the same input sequence using different learned projections (W_Q, W_K, W_V).'
    },
];

export default function PracticePanel() {
    const [mode, setMode] = useState('quiz'); // 'quiz' or 'calculator'
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showExplanation, setShowExplanation] = useState(false);

    // Calculator state
    const [calcQ, setCalcQ] = useState([1, 0, 0, 1]);
    const [calcK, setCalcK] = useState([1, 0, 1, 0]);
    const [calcV, setCalcV] = useState([0.5, 0.5, 0.5, 0.5]);
    const [calcDk, setCalcDk] = useState(4);

    const handleAnswer = (optionIndex) => {
        setSelectedAnswer(optionIndex);
        setShowExplanation(true);
        setAnswers({ ...answers, [currentQuestion]: optionIndex });
    };

    const nextQuestion = () => {
        if (currentQuestion < quizQuestions.length - 1) {
            setCurrentQuestion(currentQuestion + 1);
            setSelectedAnswer(null);
            setShowExplanation(false);
        } else {
            setShowResults(true);
        }
    };

    const resetQuiz = () => {
        setCurrentQuestion(0);
        setAnswers({});
        setShowResults(false);
        setSelectedAnswer(null);
        setShowExplanation(false);
    };

    const getScore = () => {
        let correct = 0;
        Object.entries(answers).forEach(([q, a]) => {
            if (quizQuestions[parseInt(q)].correct === a) correct++;
        });
        return correct;
    };

    // Calculator functions
    const dotProduct = calcQ.reduce((sum, val, i) => sum + val * calcK[i], 0);
    const scaledScore = dotProduct / Math.sqrt(calcDk);
    const attention = Math.exp(scaledScore) / (Math.exp(scaledScore) + 1); // Simplified softmax with 2 keys
    const output = calcV.map(v => (v * attention).toFixed(3));

    const q = quizQuestions[currentQuestion];

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-4xl mx-auto">
                {/* Mode Toggle */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setMode('quiz')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                            mode === 'quiz'
                                ? 'bg-gradient-to-r from-rose-500 to-red-500 text-white shadow-lg'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                    >
                        <Brain size={20} />
                        Quiz ({quizQuestions.length} Questions)
                    </button>
                    <button
                        onClick={() => setMode('calculator')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                            mode === 'calculator'
                                ? 'bg-gradient-to-r from-rose-500 to-red-500 text-white shadow-lg'
                                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                    >
                        <Calculator size={20} />
                        Interactive Calculator
                    </button>
                </div>

                {mode === 'quiz' ? (
                    // Quiz Mode
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                        {!showResults ? (
                            <>
                                {/* Progress */}
                                <div className="flex justify-between items-center mb-6">
                                    <span className="text-slate-400">
                                        Question {currentQuestion + 1} of {quizQuestions.length}
                                    </span>
                                    <div className="flex gap-1">
                                        {quizQuestions.map((_, i) => (
                                            <div
                                                key={i}
                                                className={`w-3 h-3 rounded-full ${
                                                    i < currentQuestion ? 'bg-green-500' :
                                                    i === currentQuestion ? 'bg-rose-500' : 'bg-slate-600'
                                                }`}
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Question */}
                                <h3 className="text-xl font-bold text-white mb-6">{q.question}</h3>

                                {/* Options */}
                                <div className="space-y-3 mb-6">
                                    {q.options.map((option, i) => {
                                        const isSelected = selectedAnswer === i;
                                        const isCorrect = q.correct === i;
                                        const showCorrectness = showExplanation;

                                        return (
                                            <button
                                                key={i}
                                                onClick={() => !showExplanation && handleAnswer(i)}
                                                disabled={showExplanation}
                                                className={`w-full text-left p-4 rounded-xl border transition-all ${
                                                    showCorrectness
                                                        ? isCorrect
                                                            ? 'bg-green-500/20 border-green-500 text-green-300'
                                                            : isSelected
                                                                ? 'bg-red-500/20 border-red-500 text-red-300'
                                                                : 'bg-slate-700/50 border-slate-600 text-slate-400'
                                                        : isSelected
                                                            ? 'bg-rose-500/20 border-rose-500 text-white'
                                                            : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:bg-slate-600/50'
                                                }`}
                                            >
                                                <div className="flex items-center gap-3">
                                                    {showCorrectness && isCorrect && (
                                                        <CheckCircle2 className="text-green-500 flex-shrink-0" size={20} />
                                                    )}
                                                    {showCorrectness && isSelected && !isCorrect && (
                                                        <XCircle className="text-red-500 flex-shrink-0" size={20} />
                                                    )}
                                                    <span>{option}</span>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>

                                {/* Explanation */}
                                {showExplanation && (
                                    <div className="bg-amber-500/10 p-4 rounded-xl border border-amber-500/30 mb-4">
                                        <div className="flex items-start gap-2">
                                            <Lightbulb className="text-amber-400 flex-shrink-0 mt-1" size={18} />
                                            <p className="text-amber-200">{q.explanation}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Next Button */}
                                {showExplanation && (
                                    <button
                                        onClick={nextQuestion}
                                        className="w-full py-3 bg-gradient-to-r from-rose-500 to-red-500 text-white rounded-xl font-medium hover:opacity-90"
                                    >
                                        {currentQuestion < quizQuestions.length - 1 ? 'Next Question' : 'See Results'}
                                    </button>
                                )}
                            </>
                        ) : (
                            // Results
                            <div className="text-center py-8">
                                <div className="text-6xl mb-4">
                                    {getScore() === quizQuestions.length ? '🏆' : 
                                     getScore() >= quizQuestions.length * 0.7 ? '⭐' : 
                                     getScore() >= quizQuestions.length * 0.5 ? '👍' : '📚'}
                                </div>
                                <h3 className="text-2xl font-bold text-white mb-2">
                                    {getScore() === quizQuestions.length ? 'Perfect Score!' :
                                     getScore() >= quizQuestions.length * 0.7 ? 'Great Job!' :
                                     getScore() >= quizQuestions.length * 0.5 ? 'Good Effort!' : 'Keep Learning!'}
                                </h3>
                                <p className="text-xl text-slate-400 mb-6">
                                    You scored <span className="font-bold text-rose-400">{getScore()}</span> out of {quizQuestions.length}
                                </p>
                                <button
                                    onClick={resetQuiz}
                                    className="flex items-center gap-2 mx-auto px-6 py-3 bg-gradient-to-r from-rose-500 to-red-500 text-white rounded-xl hover:opacity-90"
                                >
                                    <RotateCcw size={18} />
                                    Try Again
                                </button>
                            </div>
                        )}
                    </div>
                ) : (
                    // Calculator Mode
                    <div className="space-y-6">
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                <Zap className="text-yellow-400" size={24} />
                                Attention Score Calculator
                            </h3>
                            
                            <div className="grid grid-cols-2 gap-6">
                                {/* Query Input */}
                                <div>
                                    <label className="block text-blue-400 font-medium mb-2">Query (Q)</label>
                                    <div className="flex gap-2">
                                        {calcQ.map((val, i) => (
                                            <input
                                                key={i}
                                                type="number"
                                                step="0.1"
                                                value={val}
                                                onChange={(e) => {
                                                    const newQ = [...calcQ];
                                                    newQ[i] = parseFloat(e.target.value) || 0;
                                                    setCalcQ(newQ);
                                                }}
                                                className="w-16 px-2 py-2 bg-blue-500/20 border border-blue-500/50 rounded-lg text-blue-300 text-center font-mono"
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Key Input */}
                                <div>
                                    <label className="block text-green-400 font-medium mb-2">Key (K)</label>
                                    <div className="flex gap-2">
                                        {calcK.map((val, i) => (
                                            <input
                                                key={i}
                                                type="number"
                                                step="0.1"
                                                value={val}
                                                onChange={(e) => {
                                                    const newK = [...calcK];
                                                    newK[i] = parseFloat(e.target.value) || 0;
                                                    setCalcK(newK);
                                                }}
                                                className="w-16 px-2 py-2 bg-green-500/20 border border-green-500/50 rounded-lg text-green-300 text-center font-mono"
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Value Input */}
                                <div>
                                    <label className="block text-purple-400 font-medium mb-2">Value (V)</label>
                                    <div className="flex gap-2">
                                        {calcV.map((val, i) => (
                                            <input
                                                key={i}
                                                type="number"
                                                step="0.1"
                                                value={val}
                                                onChange={(e) => {
                                                    const newV = [...calcV];
                                                    newV[i] = parseFloat(e.target.value) || 0;
                                                    setCalcV(newV);
                                                }}
                                                className="w-16 px-2 py-2 bg-purple-500/20 border border-purple-500/50 rounded-lg text-purple-300 text-center font-mono"
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Dimension */}
                                <div>
                                    <label className="block text-orange-400 font-medium mb-2">Dimension (d_k)</label>
                                    <select
                                        value={calcDk}
                                        onChange={(e) => setCalcDk(Number(e.target.value))}
                                        className="px-4 py-2 bg-orange-500/20 border border-orange-500/50 rounded-lg text-orange-300"
                                    >
                                        <option value={4}>4</option>
                                        <option value={8}>8</option>
                                        <option value={16}>16</option>
                                        <option value={64}>64</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Computation Steps */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                            <h4 className="text-lg font-bold text-white mb-4">Step-by-Step Computation</h4>
                            
                            <div className="space-y-4">
                                {/* Step 1: Dot Product */}
                                <div className="bg-slate-700/50 rounded-xl p-4">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <span className="text-slate-400">Step 1: </span>
                                            <span className="text-white">Q · K = </span>
                                            <span className="text-slate-400 font-mono">
                                                {calcQ.map((q, i) => `${q}×${calcK[i]}`).join(' + ')}
                                            </span>
                                        </div>
                                        <div className="bg-blue-500/30 px-4 py-2 rounded-lg">
                                            <span className="text-blue-300 font-mono text-lg">{dotProduct.toFixed(3)}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Step 2: Scale */}
                                <div className="bg-slate-700/50 rounded-xl p-4">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <span className="text-slate-400">Step 2: </span>
                                            <span className="text-white">score / √d_k = </span>
                                            <span className="text-slate-400 font-mono">
                                                {dotProduct.toFixed(3)} / √{calcDk}
                                            </span>
                                        </div>
                                        <div className="bg-orange-500/30 px-4 py-2 rounded-lg">
                                            <span className="text-orange-300 font-mono text-lg">{scaledScore.toFixed(3)}</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Step 3: Softmax (simplified) */}
                                <div className="bg-slate-700/50 rounded-xl p-4">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <span className="text-slate-400">Step 3: </span>
                                            <span className="text-white">softmax (simplified) = </span>
                                            <span className="text-slate-400 font-mono">
                                                σ({scaledScore.toFixed(3)})
                                            </span>
                                        </div>
                                        <div className="bg-green-500/30 px-4 py-2 rounded-lg">
                                            <span className="text-green-300 font-mono text-lg">{(attention * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Step 4: Weighted Value */}
                                <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-purple-500/30">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <span className="text-slate-400">Step 4: </span>
                                            <span className="text-white">attention × V = </span>
                                            <span className="text-slate-400 font-mono">
                                                {attention.toFixed(3)} × V
                                            </span>
                                        </div>
                                        <div className="flex gap-2">
                                            {output.map((val, i) => (
                                                <div key={i} className="bg-purple-500/30 px-3 py-2 rounded-lg">
                                                    <span className="text-purple-300 font-mono">{val}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Visual Representation */}
                        <div className="bg-amber-500/10 rounded-xl p-4 border border-amber-500/30">
                            <div className="flex items-start gap-2">
                                <Lightbulb className="text-amber-400 flex-shrink-0 mt-1" size={18} />
                                <div className="text-amber-200 text-sm">
                                    <strong>Experiment!</strong> Try making Q and K more similar (higher dot product) 
                                    or more different (lower dot product) and see how the attention weight changes. 
                                    Notice how the scaled score prevents extreme values.
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Summary Card */}
                <div className="mt-8 bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl p-6 border border-slate-600">
                    <h4 className="text-white font-bold mb-4 text-center">🎓 Key Takeaways</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="space-y-2">
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Attention = soft, differentiable lookup</span>
                            </div>
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Q, K, V are learned projections</span>
                            </div>
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Scaling prevents gradient issues</span>
                            </div>
                        </div>
                        <div className="space-y-2">
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Multi-head = diverse perspectives</span>
                            </div>
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Output = weighted sum of Values</span>
                            </div>
                            <div className="flex items-start gap-2">
                                <span className="text-green-400">✓</span>
                                <span className="text-slate-300">Foundation of all Transformers!</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
