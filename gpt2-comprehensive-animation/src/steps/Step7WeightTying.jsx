import React, { useState } from 'react';

export default function Step7WeightTying({ onComplete, onNext, onPrev }) {
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const vocabSize = 50257;
    const d_model = 768;
    const params = (vocabSize * d_model) / 1000000; // in millions

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('memory') || quizAnswer.toLowerCase().includes('parameter') || quizAnswer.toLowerCase().includes('size');
        setQuizFeedback(correct
            ? '✓ Correct! Weight tying significantly reduces the total number of parameters, saving memory and improving regularization.'
            : '✗ Try again. What is the main benefit of reusing the same matrix for two different parts of the model?'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 7: Weight Tying</h2>
                <p className="text-gray-400">A clever trick to save parameters</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">What is Weight Tying?</h3>
                <p className="text-gray-300">
                    In GPT-2, the matrix used to convert tokens to embeddings (Input Embedding) is <strong>the same matrix</strong> used to convert the final output back to token probabilities (Unembedding / Output Projection).
                </p>
                <div className="bg-gray-900 p-4 rounded font-mono text-center text-emerald-400">
                    W<sub>input</sub> = W<sub>output</sub><sup>T</sup>
                </div>
            </div>

            {/* Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-6">
                <h3 className="text-xl font-semibold text-emerald-400">Parameter Savings</h3>

                <div className="flex flex-col md:flex-row gap-8 items-center justify-center">
                    {/* Without Tying */}
                    <div className="bg-gray-900 p-6 rounded-lg border border-red-900/50 w-full md:w-1/2">
                        <h4 className="text-red-400 font-bold mb-4 text-center">Without Tying</h4>
                        <div className="space-y-2 text-sm text-gray-300">
                            <div className="flex justify-between">
                                <span>Input Embeddings:</span>
                                <span>{params.toFixed(1)}M</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Output Projection:</span>
                                <span>{params.toFixed(1)}M</span>
                            </div>
                            <div className="border-t border-gray-700 pt-2 flex justify-between font-bold text-white">
                                <span>Total Cost:</span>
                                <span>{(params * 2).toFixed(1)}M</span>
                            </div>
                        </div>
                    </div>

                    {/* With Tying */}
                    <div className="bg-gray-900 p-6 rounded-lg border border-emerald-900/50 w-full md:w-1/2 relative overflow-hidden">
                        <div className="absolute top-0 right-0 bg-emerald-600 text-white text-xs px-2 py-1">GPT-2</div>
                        <h4 className="text-emerald-400 font-bold mb-4 text-center">With Tying</h4>
                        <div className="space-y-2 text-sm text-gray-300">
                            <div className="flex justify-between">
                                <span>Shared Matrix:</span>
                                <span>{params.toFixed(1)}M</span>
                            </div>
                            <div className="flex justify-between opacity-50">
                                <span>(Reused):</span>
                                <span>0M</span>
                            </div>
                            <div className="border-t border-gray-700 pt-2 flex justify-between font-bold text-white">
                                <span>Total Cost:</span>
                                <span>{params.toFixed(1)}M</span>
                            </div>
                        </div>
                    </div>
                </div>

                <p className="text-center text-gray-400 text-sm">
                    For GPT-2 Small (124M params), the embedding matrix accounts for ~30% of all parameters!
                </p>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-400">📝 Exercise</h3>
                <p className="text-gray-300">
                    What is the primary benefit of weight tying in language models?
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
                    Next: Training Optimizations →
                </button>
            </div>
        </div>
    );
}
