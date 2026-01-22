import React, { useState } from 'react';

export default function Step5Norm({ onComplete, onNext, onPrev }) {
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('gradient') || quizAnswer.toLowerCase().includes('vanish');
        setQuizFeedback(correct
            ? '✓ Correct! Residual connections provide a direct path for gradients to flow backward, solving the vanishing gradient problem.'
            : '✗ Try again. What major problem in deep networks do skip connections solve?'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 5: Layer Norm & Residuals</h2>
                <p className="text-gray-800 dark:text-gray-400">The secret sauce for training deep networks</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Residual Connections (Skip Connections)</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    Deep networks are hard to train because gradients vanish as they propagate back through many layers.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                    <strong>Residual connections</strong> solve this by adding the input of a layer to its output:
                </p>
                <div className="bg-gray-900 p-4 rounded font-mono text-center text-emerald-600 dark:text-emerald-400">
                    Output = Layer(Input) + Input
                </div>
                <p className="text-gray-700 dark:text-sm">
                    This creates a "highway" for gradients to flow unchanged during backpropagation.
                </p>
            </div>

            {/* Diagram */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4 flex flex-col items-center">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Pre-Layer Normalization (GPT-2 Style)</h3>
                <div className="relative w-64 h-96 bg-gray-900 rounded-lg p-4 flex flex-col items-center justify-between border border-gray-700">
                    {/* Input */}
                    <div className="w-full text-center text-gray-800 dark:text-gray-400">Input x</div>

                    {/* Path Split */}
                    <div className="w-0.5 h-4 bg-gray-500"></div>
                    <div className="w-full h-0.5 bg-gray-500 relative">
                        <div className="absolute -right-2 -top-1 text-xs text-gray-700 dark:text-gray-500">Residual Path</div>
                    </div>

                    <div className="flex w-full justify-between">
                        {/* Main Path */}
                        <div className="flex flex-col items-center w-1/2 border-r border-gray-800 pr-2">
                            <div className="w-0.5 h-4 bg-gray-500"></div>
                            <div className="px-3 py-2 bg-purple-900 rounded border border-purple-700 text-xs text-center w-full">Layer Norm</div>
                            <div className="w-0.5 h-4 bg-gray-500"></div>
                            <div className="px-3 py-4 bg-blue-900 rounded border border-blue-700 text-xs text-center w-full font-bold">Attention / FFN</div>
                            <div className="w-0.5 h-4 bg-gray-500"></div>
                        </div>

                        {/* Skip Path Visual */}
                        <div className="w-1/2 h-full border-l-2 border-dashed border-gray-600 ml-2 relative">
                        </div>
                    </div>

                    {/* Add */}
                    <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center border border-gray-500 z-10 -mt-4">
                        +
                    </div>

                    {/* Output */}
                    <div className="w-0.5 h-4 bg-gray-500"></div>
                    <div className="w-full text-center text-gray-800 dark:text-gray-400">Output</div>
                </div>
                <p className="text-sm text-gray-800 dark:text-gray-400 mt-2 text-center">
                    In GPT-2, Layer Norm is applied <strong>before</strong> the sub-layer (Pre-LN), unlike the original Transformer (Post-LN). This improves stability.
                </p>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">📝 Exercise</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    Why are residual connections critical for training deep networks like GPT-2 (which has up to 48 layers)?
                </p>
                <textarea
                    value={quizAnswer}
                    onChange={(e) => setQuizAnswer(e.target.value)}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none h-24"
                    placeholder="Your answer..."
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
                    Next: Full Architecture →
                </button>
            </div>
        </div>
    );
}
