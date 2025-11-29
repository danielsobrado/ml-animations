import React, { useState } from 'react';

export default function Step8Training({ onComplete, onNext, onPrev }) {
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('memory') || quizAnswer.toLowerCase().includes('ram') || quizAnswer.toLowerCase().includes('vram');
        setQuizFeedback(correct
            ? '✓ Correct! Gradient accumulation allows training with large effective batch sizes even on GPUs with limited memory.'
            : '✗ Try again. What resource constraint does gradient accumulation help overcome?'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 8: Training Optimizations</h2>
                <p className="text-gray-400">How to train massive models efficiently</p>
            </div>

            {/* Optimization Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                {/* Gradient Accumulation */}
                <div className="bg-gray-800 rounded-lg p-6 space-y-3 border border-gray-700">
                    <h3 className="text-lg font-semibold text-emerald-400">Gradient Accumulation</h3>
                    <p className="text-sm text-gray-300">
                        Simulates a larger batch size by running multiple small batches and summing their gradients before updating weights.
                    </p>
                    <div className="bg-gray-900 p-3 rounded text-xs font-mono text-gray-400">
                        For i in range(accumulation_steps):<br />
                        &nbsp;&nbsp;loss = model(batch)<br />
                        &nbsp;&nbsp;loss.backward()<br />
                        optimizer.step()<br />
                        optimizer.zero_grad()
                    </div>
                </div>

                {/* Mixed Precision */}
                <div className="bg-gray-800 rounded-lg p-6 space-y-3 border border-gray-700">
                    <h3 className="text-lg font-semibold text-emerald-400">Mixed Precision (FP16)</h3>
                    <p className="text-sm text-gray-300">
                        Uses 16-bit floating point numbers for activations and gradients to save memory and speed up computation, while keeping a 32-bit copy of weights for stability.
                    </p>
                    <div className="flex gap-2 text-xs">
                        <span className="bg-blue-900 text-blue-200 px-2 py-1 rounded">2x Faster</span>
                        <span className="bg-green-900 text-green-200 px-2 py-1 rounded">50% Memory</span>
                    </div>
                </div>

                {/* Gradient Clipping */}
                <div className="bg-gray-800 rounded-lg p-6 space-y-3 border border-gray-700">
                    <h3 className="text-lg font-semibold text-emerald-400">Gradient Clipping</h3>
                    <p className="text-sm text-gray-300">
                        Caps the norm of gradient vectors to prevent "exploding gradients" which can destabilize training.
                    </p>
                    <div className="bg-gray-900 p-3 rounded text-xs font-mono text-gray-400">
                        torch.nn.utils.clip_grad_norm_(<br />
                        &nbsp;&nbsp;model.parameters(), max_norm=1.0<br />
                        )
                    </div>
                </div>

                {/* Learning Rate Schedule */}
                <div className="bg-gray-800 rounded-lg p-6 space-y-3 border border-gray-700">
                    <h3 className="text-lg font-semibold text-emerald-400">Cosine Decay with Warmup</h3>
                    <p className="text-sm text-gray-300">
                        Linearly increases LR from 0 to max (warmup), then decreases it following a cosine curve. Helps stability at start and convergence at end.
                    </p>
                    {/* Simple SVG Chart */}
                    <svg viewBox="0 0 100 40" className="w-full h-16 bg-gray-900 rounded">
                        <path d="M0,40 L10,5 Q50,5 100,40" fill="none" stroke="#10b981" strokeWidth="2" />
                    </svg>
                </div>

            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-400">📝 Exercise</h3>
                <p className="text-gray-300">
                    If your GPU runs out of memory (OOM) with batch size 32, but you want the training stability of batch size 32, which technique should you use?
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
                    Next: Inference Optimizations →
                </button>
            </div>
        </div>
    );
}
