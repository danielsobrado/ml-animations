import React, { useState } from 'react';

export default function Step6Architecture({ onComplete, onNext, onPrev }) {
    const [modelSize, setModelSize] = useState('small'); // small, medium, large, xl
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const configs = {
        small: { layers: 12, heads: 12, d_model: 768, params: '124M' },
        medium: { layers: 24, heads: 16, d_model: 1024, params: '355M' },
        large: { layers: 36, heads: 20, d_model: 1280, params: '774M' },
        xl: { layers: 48, heads: 25, d_model: 1600, params: '1.5B' },
    };

    const currentConfig = configs[modelSize];

    const checkQuiz = () => {
        const correct = quizAnswer.includes('124') || quizAnswer.toLowerCase().includes('million');
        setQuizFeedback(correct
            ? '✓ Correct! GPT-2 Small has approximately 124 million parameters.'
            : '✗ Try again. Look at the configuration table above for GPT-2 Small.'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 6: Full Architecture</h2>
                <p className="text-gray-800 dark:text-gray-400">Stacking it all together</p>
            </div>

            {/* Interactive Configurator */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-6">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">GPT-2 Configurations</h3>

                <div className="flex gap-4 mb-4">
                    {Object.keys(configs).map(size => (
                        <button
                            key={size}
                            onClick={() => setModelSize(size)}
                            className={`px-4 py-2 rounded font-bold capitalize transition-colors ${modelSize === size
                                    ? 'bg-emerald-600 text-white'
                                    : 'bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-600'
                                }`}
                        >
                            {size}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-900 p-4 rounded text-center">
                        <div className="text-gray-800 dark:text-sm">Layers</div>
                        <div className="text-2xl font-mono text-emerald-600 dark:text-emerald-400">{currentConfig.layers}</div>
                    </div>
                    <div className="bg-gray-900 p-4 rounded text-center">
                        <div className="text-gray-800 dark:text-sm">Heads</div>
                        <div className="text-2xl font-mono text-emerald-600 dark:text-emerald-400">{currentConfig.heads}</div>
                    </div>
                    <div className="bg-gray-900 p-4 rounded text-center">
                        <div className="text-gray-800 dark:text-sm">Embedding Dim</div>
                        <div className="text-2xl font-mono text-emerald-600 dark:text-emerald-400">{currentConfig.d_model}</div>
                    </div>
                    <div className="bg-gray-900 p-4 rounded text-center">
                        <div className="text-gray-800 dark:text-sm">Parameters</div>
                        <div className="text-2xl font-mono text-emerald-600 dark:text-emerald-400">{currentConfig.params}</div>
                    </div>
                </div>
            </div>

            {/* Architecture Stack Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4 flex flex-col items-center">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">The Stack</h3>
                <div className="w-64 border-2 border-gray-700 rounded-lg p-2 bg-gray-900">
                    {/* Output Head */}
                    <div className="h-10 bg-red-900/50 border border-red-700 rounded mb-2 flex items-center justify-center text-xs text-red-200">
                        Output Projection (Unembedding)
                    </div>

                    {/* Stack */}
                    <div className="space-y-1 mb-2 relative">
                        <div className="h-12 bg-blue-900/50 border border-blue-700 rounded flex items-center justify-center text-xs text-blue-200">
                            Transformer Block {currentConfig.layers}
                        </div>
                        <div className="h-8 flex items-center justify-center text-gray-700 dark:text-xs">
                            ... x {currentConfig.layers - 2} ...
                        </div>
                        <div className="h-12 bg-blue-900/50 border border-blue-700 rounded flex items-center justify-center text-xs text-blue-200">
                            Transformer Block 1
                        </div>

                        {/* Arrow */}
                        <div className="absolute -left-8 top-0 bottom-0 w-4 border-l-2 border-gray-600 flex items-center">
                            <span className="text-xs text-gray-700 dark:text-gray-500 -rotate-90 whitespace-nowrap -ml-8">Repeat {currentConfig.layers}x</span>
                        </div>
                    </div>

                    {/* Input */}
                    <div className="h-10 bg-green-900/50 border border-green-700 rounded flex items-center justify-center text-xs text-green-200">
                        Token + Pos Embeddings
                    </div>
                </div>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">📝 Exercise</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    How many parameters does the GPT-2 Small model have?
                </p>
                <input
                    type="text"
                    value={quizAnswer}
                    onChange={(e) => setQuizAnswer(e.target.value)}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    placeholder="e.g., 100M"
                />
                <button
                    onClick={checkQuiz}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-semibold transition-colors"
                >
                    Check Answer
                </button>
                {quizFeedback && (
                    <div className={`p-3 rounded ${quizFeedback.startsWith('✓') ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                        {quizFeedback}
                    </div>
                )}
            </div>

            {/* Navigation */}
            <div className="flex justify-between">
                <button
                    onClick={onPrev}
                    className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded font-semibold transition-colors"
                >
                    ← Previous
                </button>
                <button
                    onClick={onNext}
                    className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 rounded font-semibold transition-colors"
                >
                    Next: Weight Tying →
                </button>
            </div>
        </div>
    );
}
