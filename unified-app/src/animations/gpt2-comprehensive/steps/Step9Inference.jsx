import React, { useState } from 'react';

export default function Step9Inference({ onComplete, onPrev }) {
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('compute') || quizAnswer.toLowerCase().includes('calculation') || quizAnswer.toLowerCase().includes('redundant');
        setQuizFeedback(correct
            ? '✓ Correct! KV Caching prevents re-computing Key and Value vectors for tokens we have already processed.'
            : '✗ Try again. Why is it wasteful to re-process the entire prompt for every new token generated?'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 9: Inference Optimizations</h2>
                <p className="text-gray-800 dark:text-gray-400">Making generation fast</p>
            </div>

            {/* KV Cache Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">KV Caching</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    When generating text token-by-token, we don't need to recompute the Keys and Values for previous tokens. We can <strong>cache</strong> them.
                </p>

                <div className="flex flex-col gap-4 mt-4">
                    {/* Without Cache */}
                    <div className="bg-gray-900 p-4 rounded border border-red-900/30">
                        <div className="text-red-400 font-bold text-sm mb-2">Without Cache (O(N²) complexity)</div>
                        <div className="flex gap-1 text-xs font-mono text-gray-700 dark:text-gray-500">
                            <span className="bg-gray-700 px-1 rounded text-white">The</span>
                            <span className="bg-gray-700 px-1 rounded text-white">cat</span>
                            <span className="bg-gray-700 px-1 rounded text-white">sat</span>
                            <span className="bg-emerald-600 px-1 rounded text-white">on</span>
                        </div>
                        <div className="mt-1 text-xs text-gray-800 dark:text-gray-400">Recomputes attention for "The", "cat", "sat"</div>
                    </div>

                    {/* With Cache */}
                    <div className="bg-gray-900 p-4 rounded border border-emerald-900/30">
                        <div className="text-emerald-600 dark:text-emerald-400 font-bold text-sm mb-2">With Cache (O(N) complexity)</div>
                        <div className="flex gap-1 text-xs font-mono text-gray-700 dark:text-gray-500">
                            <span className="bg-gray-800 px-1 rounded text-gray-800 dark:text-gray-400 border border-emerald-500/50">The</span>
                            <span className="bg-gray-800 px-1 rounded text-gray-800 dark:text-gray-400 border border-emerald-500/50">cat</span>
                            <span className="bg-gray-800 px-1 rounded text-gray-800 dark:text-gray-400 border border-emerald-500/50">sat</span>
                            <span className="bg-emerald-600 px-1 rounded text-white">on</span>
                        </div>
                        <div className="mt-1 text-xs text-gray-800 dark:text-gray-400">Uses cached K/V for "The", "cat", "sat". Only computes "on".</div>
                    </div>
                </div>
            </div>

            {/* Sampling Strategies */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Sampling Strategies</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    How do we pick the next token from the probabilities?
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-900 p-4 rounded">
                        <div className="font-bold text-white mb-1">Greedy</div>
                        <div className="text-xs text-gray-800 dark:text-gray-400">Always pick the highest probability token. Can be repetitive.</div>
                    </div>
                    <div className="bg-gray-900 p-4 rounded">
                        <div className="font-bold text-white mb-1">Top-k</div>
                        <div className="text-xs text-gray-800 dark:text-gray-400">Sample from the top <em>k</em> most likely tokens.</div>
                    </div>
                    <div className="bg-gray-900 p-4 rounded">
                        <div className="font-bold text-white mb-1">Nucleus (Top-p)</div>
                        <div className="text-xs text-gray-800 dark:text-gray-400">Sample from the smallest set of tokens whose cumulative probability exceeds <em>p</em>.</div>
                    </div>
                </div>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">📝 Exercise</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    Why does KV Caching make text generation significantly faster?
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

            {/* Completion */}
            <div className="bg-emerald-900 bg-opacity-30 border border-emerald-700 rounded-lg p-8 text-center space-y-4 mt-8">
                <h2 className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">🎉 Course Complete!</h2>
                <p className="text-gray-700 dark:text-gray-300">
                    You've explored the architecture and optimizations of GPT-2.
                </p>
                <button
                    onClick={() => window.location.href = '/'}
                    className="px-8 py-3 bg-emerald-600 hover:bg-emerald-700 rounded font-bold transition-colors"
                >
                    Start Over
                </button>
            </div>

            {/* Navigation */}
            <div className="flex justify-start">
                <button
                    onClick={onPrev}
                    className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded font-semibold transition-colors"
                >
                    ← Previous
                </button>
            </div>
        </div>
    );
}
