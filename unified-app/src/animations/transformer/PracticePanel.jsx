import React, { useState } from 'react';
import { CheckCircle, XCircle, RotateCcw, Trophy, Calculator, Lightbulb } from 'lucide-react';

export default function PracticePanel() {
    const [mode, setMode] = useState('quiz');
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showResult, setShowResult] = useState(false);
    const [score, setScore] = useState(0);
    const [answered, setAnswered] = useState([]);

    // Calculator state
    const [seqLen, setSeqLen] = useState(128);
    const [dModel, setDModel] = useState(512);
    const [numHeads, setNumHeads] = useState(8);
    const [numLayers, setNumLayers] = useState(6);
    const [dFF, setDFF] = useState(2048);
    const [vocabSize, setVocabSize] = useState(32000);

    const questions = [
        {
            question: 'What are the three main components that the encoder produces from the input?',
            options: [
                'Input, Output, Hidden',
                'Query, Key, Value',
                'Attention, FFN, Norm',
                'Embedding, Positional, Output'
            ],
            correct: 1,
            explanation: 'The input is projected into Query (Q), Key (K), and Value (V) vectors for attention computation.'
        },
        {
            question: 'Why is scaling by √d_k important in attention?',
            options: [
                'To make the model faster',
                'To prevent dot products from becoming too large and softmax from saturating',
                'To reduce memory usage',
                'To enable multi-head attention'
            ],
            correct: 1,
            explanation: 'Large dot products push softmax into regions with very small gradients, causing vanishing gradients.'
        },
        {
            question: 'What is the purpose of the causal mask in the decoder?',
            options: [
                'To speed up training',
                'To reduce memory usage',
                'To prevent attending to future tokens during generation',
                'To improve attention scores'
            ],
            correct: 2,
            explanation: 'The causal mask ensures that position i can only attend to positions 0 through i, preventing "cheating" by looking at future tokens.'
        },
        {
            question: 'In cross-attention, where do Q, K, and V come from?',
            options: [
                'All from encoder',
                'All from decoder',
                'Q from decoder, K and V from encoder',
                'Q and K from encoder, V from decoder'
            ],
            correct: 2,
            explanation: 'Cross-attention allows the decoder to "query" the encoder\'s representation. Q comes from decoder, K and V from encoder output.'
        },
        {
            question: 'What is the typical relationship between d_model and d_ff?',
            options: [
                'd_ff = d_model',
                'd_ff = 2 × d_model',
                'd_ff = 4 × d_model',
                'd_ff = d_model / 2'
            ],
            correct: 2,
            explanation: 'The feed-forward hidden dimension is typically 4× the model dimension (e.g., 512 → 2048).'
        },
        {
            question: 'Which transformer variant is BERT?',
            options: [
                'Decoder-only',
                'Encoder-only',
                'Encoder-decoder',
                'Neither'
            ],
            correct: 1,
            explanation: 'BERT uses only the encoder with bidirectional attention, trained with masked language modeling.'
        },
        {
            question: 'Which transformer variant is GPT?',
            options: [
                'Decoder-only',
                'Encoder-only',
                'Encoder-decoder',
                'Neither'
            ],
            correct: 0,
            explanation: 'GPT uses only the decoder with causal attention, trained with next token prediction.'
        },
        {
            question: 'What does "multi-head" attention allow the model to do?',
            options: [
                'Process multiple sentences at once',
                'Attend to information from different representation subspaces',
                'Use multiple GPUs',
                'Generate multiple outputs'
            ],
            correct: 1,
            explanation: 'Multiple heads allow the model to jointly attend to information from different representation subspaces at different positions.'
        },
        {
            question: 'Why are residual connections important in transformers?',
            options: [
                'They reduce the number of parameters',
                'They make the model faster',
                'They help gradients flow and enable training deep networks',
                'They improve attention scores'
            ],
            correct: 2,
            explanation: 'Residual connections allow gradients to flow directly through the network, enabling training of very deep models.'
        },
        {
            question: 'What is the computational complexity of self-attention with respect to sequence length n?',
            options: [
                'O(n)',
                'O(n log n)',
                'O(n²)',
                'O(n³)'
            ],
            correct: 2,
            explanation: 'Self-attention computes attention between every pair of positions, resulting in O(n²) complexity. This is why long contexts are challenging!'
        }
    ];

    const handleAnswer = (index) => {
        if (showResult) return;
        setSelectedAnswer(index);
        setShowResult(true);
        if (index === questions[currentQuestion].correct) {
            setScore(score + 1);
        }
        setAnswered([...answered, currentQuestion]);
    };

    const nextQuestion = () => {
        if (currentQuestion < questions.length - 1) {
            setCurrentQuestion(currentQuestion + 1);
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

    // Parameter calculations
    const dK = dModel / numHeads;
    const attentionParams = numLayers * (3 * dModel * dModel + dModel * dModel); // Q,K,V + output projection
    const ffnParams = numLayers * (dModel * dFF + dFF * dModel); // Two linear layers
    const embeddingParams = vocabSize * dModel + seqLen * dModel; // Token + positional (simplified)
    const totalParams = attentionParams + ffnParams + embeddingParams;

    const formatNumber = (num) => {
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toString();
    };

    return (
        <div className="p-6 min-h-screen">
            <div className="max-w-4xl mx-auto">
                {/* Mode Toggle */}
                <div className="flex justify-center gap-4 mb-8">
                    <button
                        onClick={() => setMode('quiz')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                            mode === 'quiz'
                                ? 'bg-rose-500 text-white'
                                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                    >
                        <Trophy size={20} />
                        Quiz ({questions.length} Questions)
                    </button>
                    <button
                        onClick={() => setMode('calculator')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                            mode === 'calculator'
                                ? 'bg-indigo-500 text-white'
                                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                    >
                        <Calculator size={20} />
                        Parameter Calculator
                    </button>
                </div>

                {mode === 'quiz' ? (
                    <>
                        {/* Quiz Progress */}
                        <div className="flex justify-center gap-2 mb-6">
                            {questions.map((_, i) => (
                                <div
                                    key={i}
                                    className={`w-3 h-3 rounded-full transition-all ${
                                        currentQuestion === i
                                            ? 'bg-rose-500 scale-125'
                                            : answered.includes(i)
                                                ? 'bg-green-500'
                                                : 'bg-slate-600'
                                    }`}
                                />
                            ))}
                        </div>

                        {/* Score */}
                        <div className="text-center mb-6">
                            <span className="text-slate-400">Score: </span>
                            <span className="text-white font-bold">{score}</span>
                            <span className="text-slate-400"> / {answered.length}</span>
                        </div>

                        {/* Question Card */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700 mb-6">
                            <div className="text-slate-400 text-sm mb-2">
                                Question {currentQuestion + 1} of {questions.length}
                            </div>
                            <h3 className="text-white text-xl font-bold mb-6">
                                {questions[currentQuestion].question}
                            </h3>

                            <div className="space-y-3">
                                {questions[currentQuestion].options.map((option, i) => (
                                    <button
                                        key={i}
                                        onClick={() => handleAnswer(i)}
                                        disabled={showResult}
                                        className={`w-full p-4 rounded-xl text-left transition-all ${
                                            showResult
                                                ? i === questions[currentQuestion].correct
                                                    ? 'bg-green-500/20 border-2 border-green-500 text-green-300'
                                                    : selectedAnswer === i
                                                        ? 'bg-red-500/20 border-2 border-red-500 text-red-300'
                                                        : 'bg-slate-700/50 text-slate-400'
                                                : selectedAnswer === i
                                                    ? 'bg-blue-500/20 border-2 border-blue-500 text-white'
                                                    : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
                                        }`}
                                    >
                                        <div className="flex items-center gap-3">
                                            {showResult && i === questions[currentQuestion].correct && (
                                                <CheckCircle className="text-green-400" size={20} />
                                            )}
                                            {showResult && selectedAnswer === i && i !== questions[currentQuestion].correct && (
                                                <XCircle className="text-red-400" size={20} />
                                            )}
                                            <span>{option}</span>
                                        </div>
                                    </button>
                                ))}
                            </div>

                            {/* Explanation */}
                            {showResult && (
                                <div className="mt-6 p-4 bg-blue-500/10 rounded-xl border border-blue-500/30">
                                    <div className="flex items-start gap-2">
                                        <Lightbulb className="text-blue-400 mt-1" size={18} />
                                        <p className="text-slate-300 text-sm">
                                            {questions[currentQuestion].explanation}
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Navigation */}
                        <div className="flex justify-between">
                            <button
                                onClick={resetQuiz}
                                className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg"
                            >
                                <RotateCcw size={18} />
                                Reset
                            </button>
                            
                            {showResult && currentQuestion < questions.length - 1 && (
                                <button
                                    onClick={nextQuestion}
                                    className="px-6 py-2 bg-rose-500 hover:bg-rose-600 text-white rounded-lg font-medium"
                                >
                                    Next Question →
                                </button>
                            )}

                            {currentQuestion === questions.length - 1 && showResult && (
                                <div className="text-white font-bold">
                                    Final Score: {score}/{questions.length} ({Math.round(score/questions.length * 100)}%)
                                </div>
                            )}
                        </div>
                    </>
                ) : (
                    <>
                        {/* Parameter Calculator */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700 mb-6">
                            <h3 className="text-white font-bold mb-6 flex items-center gap-2">
                                <Calculator size={20} className="text-indigo-400" />
                                Transformer Parameter Calculator
                            </h3>

                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
                                <div>
                                    <label className="text-slate-400 text-sm">d_model</label>
                                    <input
                                        type="number"
                                        value={dModel}
                                        onChange={(e) => setDModel(parseInt(e.target.value) || 0)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="text-slate-400 text-sm">Num Heads</label>
                                    <input
                                        type="number"
                                        value={numHeads}
                                        onChange={(e) => setNumHeads(parseInt(e.target.value) || 1)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="text-slate-400 text-sm">Num Layers</label>
                                    <input
                                        type="number"
                                        value={numLayers}
                                        onChange={(e) => setNumLayers(parseInt(e.target.value) || 1)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="text-slate-400 text-sm">d_ff</label>
                                    <input
                                        type="number"
                                        value={dFF}
                                        onChange={(e) => setDFF(parseInt(e.target.value) || 0)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="text-slate-400 text-sm">Vocab Size</label>
                                    <input
                                        type="number"
                                        value={vocabSize}
                                        onChange={(e) => setVocabSize(parseInt(e.target.value) || 0)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                                <div>
                                    <label className="text-slate-400 text-sm">Max Seq Len</label>
                                    <input
                                        type="number"
                                        value={seqLen}
                                        onChange={(e) => setSeqLen(parseInt(e.target.value) || 0)}
                                        className="w-full mt-1 bg-slate-700 text-white p-2 rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                                    />
                                </div>
                            </div>

                            {/* Derived Values */}
                            <div className="bg-slate-700/50 p-4 rounded-lg mb-6">
                                <h4 className="text-slate-400 text-sm mb-2">Derived Values</h4>
                                <div className="flex gap-6">
                                    <div>
                                        <span className="text-slate-400">d_k = d_v = </span>
                                        <span className="text-indigo-400 font-mono">{dK}</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">d_model / h = </span>
                                        <span className="text-indigo-400 font-mono">{dModel} / {numHeads}</span>
                                    </div>
                                </div>
                            </div>

                            {/* Parameter Breakdown */}
                            <div className="space-y-3">
                                <div className="flex justify-between items-center p-3 bg-purple-500/10 rounded-lg">
                                    <span className="text-slate-300">Attention Parameters (per layer × N)</span>
                                    <span className="text-purple-400 font-mono">{formatNumber(attentionParams)}</span>
                                </div>
                                <div className="flex justify-between items-center p-3 bg-orange-500/10 rounded-lg">
                                    <span className="text-slate-300">FFN Parameters (per layer × N)</span>
                                    <span className="text-orange-400 font-mono">{formatNumber(ffnParams)}</span>
                                </div>
                                <div className="flex justify-between items-center p-3 bg-blue-500/10 rounded-lg">
                                    <span className="text-slate-300">Embedding Parameters</span>
                                    <span className="text-blue-400 font-mono">{formatNumber(embeddingParams)}</span>
                                </div>
                                <div className="flex justify-between items-center p-4 bg-green-500/20 rounded-lg border border-green-500/50">
                                    <span className="text-white font-bold">Total Parameters (approx)</span>
                                    <span className="text-green-400 font-mono text-xl">{formatNumber(totalParams)}</span>
                                </div>
                            </div>
                        </div>

                        {/* Presets */}
                        <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                            <h4 className="text-white font-bold mb-4">📚 Try Famous Model Configs</h4>
                            <div className="flex flex-wrap gap-2">
                                <button
                                    onClick={() => { setDModel(512); setNumHeads(8); setNumLayers(6); setDFF(2048); setVocabSize(37000); }}
                                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
                                >
                                    Original Transformer
                                </button>
                                <button
                                    onClick={() => { setDModel(768); setNumHeads(12); setNumLayers(12); setDFF(3072); setVocabSize(30522); }}
                                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
                                >
                                    BERT-Base
                                </button>
                                <button
                                    onClick={() => { setDModel(768); setNumHeads(12); setNumLayers(12); setDFF(3072); setVocabSize(50257); }}
                                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
                                >
                                    GPT-2 Small
                                </button>
                                <button
                                    onClick={() => { setDModel(4096); setNumHeads(32); setNumLayers(32); setDFF(11008); setVocabSize(32000); }}
                                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm"
                                >
                                    LLaMA-7B
                                </button>
                            </div>
                        </div>
                    </>
                )}

                {/* Key Takeaways */}
                <div className="mt-8 bg-amber-500/10 rounded-2xl p-6 border border-amber-500/30">
                    <h3 className="text-amber-400 font-bold mb-4 flex items-center gap-2">
                        🎓 Key Takeaways
                    </h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">Encoder: bidirectional, understanding</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">Decoder: causal, generation</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">Multi-head = multiple perspectives</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">Residuals enable deep networks</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">O(n²) attention complexity</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <span className="text-green-400">✓</span>
                            <span className="text-slate-300">Foundation of all modern LLMs!</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
