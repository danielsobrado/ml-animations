import React, { useState, useEffect, useRef } from 'react';

export default function Step2Positional({ onComplete, onNext, onPrev }) {
    const canvasRef = useRef(null);
    const [selectedPos, setSelectedPos] = useState(0);
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    const maxSeqLen = 1024; // GPT-2's max context
    const embeddingDim = 768;

    // Draw positional encoding visualization
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = '#1f2937';
        ctx.fillRect(0, 0, width, height);

        // Draw simplified positional encoding pattern
        const numPositions = 50;
        const cellWidth = width / numPositions;
        const cellHeight = 30;

        for (let pos = 0; pos < numPositions; pos++) {
            const x = pos * cellWidth;
            const isSelected = pos === selectedPos;

            // Simple visualization of encoding pattern
            const hue = (pos / numPositions) * 360;
            ctx.fillStyle = isSelected ? `hsl(${hue}, 100%, 60%)` : `hsl(${hue}, 70%, 40%)`;
            ctx.fillRect(x, 50, cellWidth - 1, cellHeight);

            if (isSelected) {
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, 50, cellWidth - 1, cellHeight);
            }
        }

        // Labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px monospace';
        ctx.fillText('Position 0', 5, 40);
        ctx.fillText(`Position ${numPositions - 1}`, width - 80, 40);
        ctx.fillText('Each position gets a unique learned vector', 5, height - 10);

    }, [selectedPos]);

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('order') || quizAnswer.toLowerCase().includes('position');
        setQuizFeedback(correct
            ? '✓ Correct! Without positional encodings, the model would treat "dog bites man" the same as "man bites dog".'
            : '✗ Try again. Think about what information is lost without position encodings.'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 2: Positional Encoding</h2>
                <p className="text-gray-800 dark:text-gray-400">Teaching the model about word order</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Why Positional Encoding?</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    Self-attention is <strong>permutation-invariant</strong> - it treats sequences as unordered sets.
                    But word order matters! "Dog bites man" ≠ "Man bites dog"
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                    GPT-2 uses <strong>learned absolute positional embeddings</strong>:
                </p>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 ml-4">
                    <li>Each position (0 to 1023) has its own learned embedding vector</li>
                    <li>These are <strong>added</strong> to the token embeddings</li>
                    <li>Allows the model to learn position-dependent patterns</li>
                </ul>
            </div>

            {/* Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Positional Embedding Visualization</h3>
                <canvas
                    ref={canvasRef}
                    width={600}
                    height={120}
                    className="border border-gray-700 rounded cursor-pointer"
                    onClick={(e) => {
                        const rect = e.target.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const pos = Math.floor((x / rect.width) * 50);
                        setSelectedPos(Math.max(0, Math.min(49, pos)));
                    }}
                />
                <div className="bg-gray-900 p-4 rounded">
                    <div className="text-sm text-gray-800 dark:text-gray-400">Selected Position: <span className="text-emerald-600 dark:text-emerald-400 font-mono">{selectedPos}</span></div>
                    <div className="text-sm text-gray-800 dark:text-gray-400">Embedding Dimension: <span className="text-emerald-600 dark:text-emerald-400 font-mono">{embeddingDim}</span></div>
                </div>
            </div>

            {/* Formula */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">How It Works</h3>
                <div className="bg-gray-900 p-4 rounded space-y-2">
                    <div className="font-mono text-sm text-gray-700 dark:text-gray-300">
                        <div>Token Embedding: <span className="text-blue-600 dark:text-blue-400">E_token</span> (768-dim)</div>
                        <div>Position Embedding: <span className="text-yellow-400">E_pos</span> (768-dim)</div>
                        <div className="mt-2 text-emerald-600 dark:text-emerald-400">Final Input: E_token + E_pos</div>
                    </div>
                </div>
                <p className="text-gray-700 dark:text-sm">
                    <strong>Alternative:</strong> Original Transformer used sinusoidal encodings (not learned).
                    GPT-2 learns them because it can capture more complex position-dependent patterns.
                </p>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">📝 Exercise</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    What would happen if we removed positional encodings from GPT-2?
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
                    Next: Multi-Head Attention →
                </button>
            </div>
        </div>
    );
}
