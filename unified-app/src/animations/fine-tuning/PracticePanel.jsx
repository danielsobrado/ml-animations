import React, { useState } from 'react';
import { CheckCircle2, XCircle, RotateCcw, Lightbulb, Layers, Zap, Settings } from 'lucide-react';

const quizQuestions = [
    {
        id: 1,
        question: 'What is the main advantage of Parameter-Efficient Fine-Tuning (PEFT)?',
        options: [
            'It makes the model more accurate',
            'It updates only a small fraction of parameters, saving memory and compute',
            'It speeds up inference time',
            'It removes the need for training data'
        ],
        correct: 1,
        explanation: 'PEFT methods like LoRA update only a tiny fraction of parameters (often <1%) while keeping the rest frozen, dramatically reducing memory and compute requirements.'
    },
    {
        id: 2,
        question: 'In LoRA, what do the matrices A and B represent?',
        options: [
            'The original weight matrix split in half',
            'Low-rank decomposition matrices that approximate the weight update',
            'Gradient accumulation buffers',
            'Activation caches'
        ],
        correct: 1,
        explanation: 'LoRA learns two small matrices B (d×r) and A (r×k) whose product BA approximates the weight update ΔW, where r is much smaller than d or k.'
    },
    {
        id: 3,
        question: 'What does the "rank" (r) parameter in LoRA control?',
        options: [
            'The learning rate',
            'The batch size',
            'The capacity/expressiveness of the adaptation',
            'The number of layers to fine-tune'
        ],
        correct: 2,
        explanation: 'The rank r controls how expressive the adaptation can be. Higher rank = more parameters = more capacity to learn complex adaptations, but also more memory.'
    },
    {
        id: 4,
        question: 'What makes QLoRA different from standard LoRA?',
        options: [
            'It uses larger rank values',
            'It quantizes the base model to 4-bit precision',
            'It removes the need for adapters',
            'It fine-tunes all parameters'
        ],
        correct: 1,
        explanation: 'QLoRA combines LoRA with 4-bit quantization of the base model (using NF4 format), reducing memory requirements by ~4x while maintaining similar performance.'
    },
    {
        id: 5,
        question: 'After training, what can be done with LoRA weights for faster inference?',
        options: [
            'They must be discarded',
            'They can be merged into the base model weights',
            'They need separate GPU memory',
            'They require retraining'
        ],
        correct: 1,
        explanation: 'LoRA weights can be merged into the base model (W\' = W + BA) after training, resulting in zero additional inference latency compared to the original model.'
    }
];

export default function PracticePanel() {
    const [mode, setMode] = useState('quiz'); // 'quiz' or 'calculator'
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [answers, setAnswers] = useState({});
    const [showResults, setShowResults] = useState(false);
    const [selectedAnswer, setSelectedAnswer] = useState(null);
    const [showExplanation, setShowExplanation] = useState(false);

    // Calculator state
    const [calcParams, setCalcParams] = useState({
        modelSize: 7, // Billion
        hiddenDim: 4096,
        numLayers: 32,
        rank: 8,
        targetModules: 2, // Number of modules (Q, V = 2; Q, K, V, O = 4)
        precision: 16, // bits
        quantization: 'none', // 'none', '8bit', '4bit'
    });

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

    // Calculate memory requirements
    const calculateMemory = () => {
        const { modelSize, hiddenDim, numLayers, rank, targetModules, precision, quantization } = calcParams;
        
        // Base model memory
        let baseModelBytes = modelSize * 1e9 * (precision / 8);
        if (quantization === '8bit') baseModelBytes = modelSize * 1e9 * 1;
        if (quantization === '4bit') baseModelBytes = modelSize * 1e9 * 0.5;
        
        // LoRA parameters per layer: 2 * hiddenDim * rank * targetModules
        const loraParamsPerLayer = 2 * hiddenDim * rank * targetModules;
        const totalLoraParams = loraParamsPerLayer * numLayers;
        const loraBytes = totalLoraParams * (precision / 8);
        
        // Optimizer states (for trainable params only)
        const optimizerBytes = loraBytes * 3; // Adam: params + momentum + variance
        
        // Gradients
        const gradientBytes = loraBytes;
        
        // Activation memory (rough estimate)
        const activationBytes = modelSize * 0.5 * 1e9;
        
        const totalTrainingBytes = baseModelBytes + loraBytes + optimizerBytes + gradientBytes + activationBytes;
        
        return {
            baseModel: (baseModelBytes / 1e9).toFixed(2),
            loraParams: totalLoraParams,
            loraMemory: (loraBytes / 1e6).toFixed(2),
            optimizer: (optimizerBytes / 1e6).toFixed(2),
            totalTraining: (totalTrainingBytes / 1e9).toFixed(2),
            percentTrainable: ((totalLoraParams / (modelSize * 1e9)) * 100).toFixed(4),
        };
    };

    const memCalc = calculateMemory();
    const q = quizQuestions[currentQuestion];

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="max-w-4xl mx-auto">
                {/* Mode Toggle */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setMode('quiz')}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${
                            mode === 'quiz'
                                ? 'bg-purple-500 text-white'
                                : 'bg-white border hover:bg-slate-50'
                        }`}
                    >
                        📝 Quiz
                    </button>
                    <button
                        onClick={() => setMode('calculator')}
                        className={`px-6 py-2 rounded-lg font-medium transition-all ${
                            mode === 'calculator'
                                ? 'bg-purple-500 text-white'
                                : 'bg-white border hover:bg-slate-50'
                        }`}
                    >
                        🧮 LoRA Calculator
                    </button>
                </div>

                {mode === 'quiz' ? (
                    // Quiz Mode
                    <div className="bg-white rounded-xl p-6 shadow-lg border">
                        {!showResults ? (
                            <>
                                {/* Progress */}
                                <div className="flex justify-between items-center mb-6">
                                    <span className="text-sm text-slate-500">
                                        Question {currentQuestion + 1} of {quizQuestions.length}
                                    </span>
                                    <div className="flex gap-1">
                                        {quizQuestions.map((_, i) => (
                                            <div
                                                key={i}
                                                className={`w-3 h-3 rounded-full ${
                                                    i < currentQuestion ? 'bg-green-500' :
                                                    i === currentQuestion ? 'bg-purple-500' : 'bg-slate-200'
                                                }`}
                                            />
                                        ))}
                                    </div>
                                </div>

                                {/* Question */}
                                <h3 className="text-xl font-bold text-slate-800 mb-6">{q.question}</h3>

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
                                                className={`w-full text-left p-4 rounded-lg border transition-all ${
                                                    showCorrectness
                                                        ? isCorrect
                                                            ? 'bg-green-100 border-green-500 text-green-800'
                                                            : isSelected
                                                                ? 'bg-red-100 border-red-500 text-red-800'
                                                                : 'bg-slate-50 border-slate-200'
                                                        : isSelected
                                                            ? 'bg-purple-100 border-purple-500'
                                                            : 'bg-white border-slate-200 hover:bg-slate-50'
                                                }`}
                                            >
                                                <div className="flex items-center gap-3">
                                                    {showCorrectness && isCorrect && (
                                                        <CheckCircle2 className="text-green-500" size={20} />
                                                    )}
                                                    {showCorrectness && isSelected && !isCorrect && (
                                                        <XCircle className="text-red-500" size={20} />
                                                    )}
                                                    <span>{option}</span>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>

                                {/* Explanation */}
                                {showExplanation && (
                                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 mb-4">
                                        <div className="flex items-start gap-2">
                                            <Lightbulb className="text-blue-500 flex-shrink-0 mt-1" size={18} />
                                            <p className="text-blue-800">{q.explanation}</p>
                                        </div>
                                    </div>
                                )}

                                {/* Next Button */}
                                {showExplanation && (
                                    <button
                                        onClick={nextQuestion}
                                        className="w-full py-3 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600"
                                    >
                                        {currentQuestion < quizQuestions.length - 1 ? 'Next Question' : 'See Results'}
                                    </button>
                                )}
                            </>
                        ) : (
                            // Results
                            <div className="text-center">
                                <div className="text-6xl mb-4">
                                    {getScore() === quizQuestions.length ? '🏆' : getScore() >= 3 ? '⭐' : '📚'}
                                </div>
                                <h3 className="text-2xl font-bold text-slate-800 mb-2">
                                    Quiz Complete!
                                </h3>
                                <p className="text-xl text-slate-600 mb-6">
                                    You scored <span className="font-bold text-purple-600">{getScore()}</span> out of {quizQuestions.length}
                                </p>
                                <button
                                    onClick={resetQuiz}
                                    className="flex items-center gap-2 mx-auto px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
                                >
                                    <RotateCcw size={18} />
                                    Try Again
                                </button>
                            </div>
                        )}
                    </div>
                ) : (
                    // Calculator Mode
                    <div className="space-y-4">
                        <div className="bg-white rounded-xl p-6 shadow-lg border">
                            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <Settings size={20} className="text-purple-500" />
                                Configure Model & LoRA
                            </h3>
                            
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        Model Size (Billion params)
                                    </label>
                                    <select
                                        value={calcParams.modelSize}
                                        onChange={(e) => setCalcParams({...calcParams, modelSize: Number(e.target.value)})}
                                        className="w-full px-3 py-2 border rounded-lg bg-white"
                                    >
                                        <option value={1}>1B</option>
                                        <option value={7}>7B</option>
                                        <option value={13}>13B</option>
                                        <option value={33}>33B</option>
                                        <option value={65}>65B</option>
                                        <option value={70}>70B</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        Hidden Dimension
                                    </label>
                                    <select
                                        value={calcParams.hiddenDim}
                                        onChange={(e) => setCalcParams({...calcParams, hiddenDim: Number(e.target.value)})}
                                        className="w-full px-3 py-2 border rounded-lg bg-white"
                                    >
                                        <option value={2048}>2048</option>
                                        <option value={4096}>4096</option>
                                        <option value={5120}>5120</option>
                                        <option value={8192}>8192</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        LoRA Rank (r)
                                    </label>
                                    <select
                                        value={calcParams.rank}
                                        onChange={(e) => setCalcParams({...calcParams, rank: Number(e.target.value)})}
                                        className="w-full px-3 py-2 border rounded-lg bg-white"
                                    >
                                        <option value={4}>4</option>
                                        <option value={8}>8</option>
                                        <option value={16}>16</option>
                                        <option value={32}>32</option>
                                        <option value={64}>64</option>
                                        <option value={128}>128</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        Target Modules
                                    </label>
                                    <select
                                        value={calcParams.targetModules}
                                        onChange={(e) => setCalcParams({...calcParams, targetModules: Number(e.target.value)})}
                                        className="w-full px-3 py-2 border rounded-lg bg-white"
                                    >
                                        <option value={2}>Q, V only (2)</option>
                                        <option value={4}>Q, K, V, O (4)</option>
                                        <option value={6}>All attention + MLP (6)</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        Number of Layers
                                    </label>
                                    <input
                                        type="number"
                                        value={calcParams.numLayers}
                                        onChange={(e) => setCalcParams({...calcParams, numLayers: Number(e.target.value)})}
                                        className="w-full px-3 py-2 border rounded-lg"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">
                                        Quantization
                                    </label>
                                    <select
                                        value={calcParams.quantization}
                                        onChange={(e) => setCalcParams({...calcParams, quantization: e.target.value})}
                                        className="w-full px-3 py-2 border rounded-lg bg-white"
                                    >
                                        <option value="none">None (FP16)</option>
                                        <option value="8bit">8-bit</option>
                                        <option value="4bit">4-bit (QLoRA)</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white rounded-xl p-6 shadow-lg border">
                            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <Zap size={20} className="text-green-500" />
                                Memory Estimates
                            </h3>
                            
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                    <div className="text-sm text-blue-600 mb-1">Base Model Memory</div>
                                    <div className="text-2xl font-bold text-blue-800">{memCalc.baseModel} GB</div>
                                </div>
                                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                    <div className="text-sm text-green-600 mb-1">LoRA Parameters</div>
                                    <div className="text-2xl font-bold text-green-800">{(memCalc.loraParams / 1e6).toFixed(2)}M</div>
                                </div>
                                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                    <div className="text-sm text-purple-600 mb-1">LoRA Memory</div>
                                    <div className="text-2xl font-bold text-purple-800">{memCalc.loraMemory} MB</div>
                                </div>
                                <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                                    <div className="text-sm text-orange-600 mb-1">% Trainable</div>
                                    <div className="text-2xl font-bold text-orange-800">{memCalc.percentTrainable}%</div>
                                </div>
                            </div>

                            <div className="mt-4 p-4 bg-slate-800 rounded-lg">
                                <div className="text-slate-400 text-sm mb-1">Estimated Training Memory</div>
                                <div className="text-3xl font-bold text-white">{memCalc.totalTraining} GB</div>
                                <div className="text-slate-400 text-sm mt-1">
                                    (includes model + LoRA + optimizer + activations)
                                </div>
                            </div>
                        </div>

                        <div className="bg-amber-50 p-4 rounded-xl border border-amber-200">
                            <h4 className="font-bold text-amber-900 mb-2 flex items-center gap-2">
                                <Lightbulb size={18} />
                                Tips
                            </h4>
                            <ul className="text-sm text-amber-800 space-y-1">
                                <li>• Start with rank=8 and increase if needed</li>
                                <li>• QLoRA (4-bit) enables training 7B models on 16GB GPUs</li>
                                <li>• More target modules = more capacity but more memory</li>
                                <li>• Actual memory may vary based on batch size and sequence length</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
