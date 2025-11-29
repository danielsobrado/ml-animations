import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

export default function Step3Attention({ onComplete, onNext, onPrev }) {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const [numHeads] = useState(12); // GPT-2 has 12 attention heads
    const [showMask, setShowMask] = useState(true);
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    // Three.js visualization of attention matrix
    useEffect(() => {
        if (!containerRef.current) return;

        const width = 400;
        const height = 400;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1f2937);
        sceneRef.current = scene;

        const camera = new THREE.OrthographicCamera(
            width / -2, width / 2, height / 2, height / -2, 0.1, 1000
        );
        camera.position.z = 100;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        // Create attention matrix visualization (simplified 5x5)
        const matrixSize = 5;
        const cellSize = 60;
        const gap = 5;
        const startX = -(matrixSize * (cellSize + gap)) / 2 + cellSize / 2;
        const startY = (matrixSize * (cellSize + gap)) / 2 - cellSize / 2;

        for (let i = 0; i < matrixSize; i++) {
            for (let j = 0; j < matrixSize; j++) {
                // Causal mask: only attend to current and previous positions
                const isMasked = showMask && (j > i);

                const geometry = new THREE.PlaneGeometry(cellSize, cellSize);
                const intensity = isMasked ? 0 : (1 - Math.abs(i - j) / matrixSize);
                const color = isMasked
                    ? new THREE.Color(0x374151)
                    : new THREE.Color().setHSL(0.55, 0.7, 0.3 + intensity * 0.4);

                const material = new THREE.MeshBasicMaterial({ color });
                const mesh = new THREE.Mesh(geometry, material);

                mesh.position.x = startX + j * (cellSize + gap);
                mesh.position.y = startY - i * (cellSize + gap);

                scene.add(mesh);

                // Add border
                const edges = new THREE.EdgesGeometry(geometry);
                const line = new THREE.LineSegments(
                    edges,
                    new THREE.LineBasicMaterial({ color: 0x4b5563 })
                );
                line.position.copy(mesh.position);
                scene.add(line);
            }
        }

        let animationId;
        const animate = () => {
            animationId = requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            cancelAnimationFrame(animationId);
            renderer.dispose();
            if (containerRef.current?.contains(renderer.domElement)) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, [showMask]);

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('future') || quizAnswer.toLowerCase().includes('next');
        setQuizFeedback(correct
            ? '✓ Correct! Causal masking prevents the model from "cheating" by looking at future tokens during training.'
            : '✗ Try again. Think about what would happen if the model could see future tokens when predicting the next word.'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 3: Multi-Head Self-Attention</h2>
                <p className="text-gray-400">The core mechanism that makes transformers powerful</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">What is Self-Attention?</h3>
                <p className="text-gray-300">
                    Self-attention allows each token to <strong>look at all other tokens</strong> in the sequence and decide how much to "pay attention" to each one.
                </p>
                <p className="text-gray-300">
                    For each token, we compute:
                </p>
                <ul className="list-disc list-inside space-y-1 text-gray-300 ml-4">
                    <li><strong>Query (Q)</strong>: What am I looking for?</li>
                    <li><strong>Key (K)</strong>: What do I contain?</li>
                    <li><strong>Value (V)</strong>: What information do I have?</li>
                </ul>
            </div>

            {/* Formula */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">The Math</h3>
                <div className="bg-gray-900 p-4 rounded space-y-2 font-mono text-sm">
                    <div className="text-gray-300">Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V</div>
                    <div className="text-gray-400 text-xs mt-2">where d<sub>k</sub> = 64 (dimension per head)</div>
                </div>
                <p className="text-gray-300 text-sm">
                    Steps: (1) Compute attention scores (QK<sup>T</sup>), (2) Scale by √d<sub>k</sub>, (3) Apply softmax, (4) Multiply by values (V)
                </p>
            </div>

            {/* Attention Matrix Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">Attention Pattern (Simplified 5×5)</h3>
                <div className="flex flex-col items-center gap-4">
                    <div
                        ref={containerRef}
                        className="border border-gray-700 rounded"
                    />
                    <div className="flex items-center gap-4">
                        <label className="flex items-center gap-2 text-gray-300">
                            <input
                                type="checkbox"
                                checked={showMask}
                                onChange={(e) => setShowMask(e.target.checked)}
                                className="w-4 h-4"
                            />
                            Show Causal Mask
                        </label>
                    </div>
                    <div className="text-sm text-gray-400 text-center">
                        Rows = Query positions, Columns = Key positions<br />
                        Brighter = Higher attention weight
                    </div>
                </div>
            </div>

            {/* Multi-Head Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-400">Multi-Head Attention</h3>
                <p className="text-gray-300">
                    GPT-2 uses <strong>{numHeads} parallel attention heads</strong>. Each head:
                </p>
                <ul className="list-disc list-inside space-y-1 text-gray-300 ml-4">
                    <li>Has its own Q, K, V weight matrices</li>
                    <li>Learns different aspects (e.g., syntax, semantics, position)</li>
                    <li>Outputs are concatenated and projected</li>
                </ul>
                <div className="bg-gray-900 p-4 rounded">
                    <div className="text-sm text-gray-400">Dimension per head: <span className="text-emerald-400">64</span> (768 / 12)</div>
                    <div className="text-sm text-gray-400">Total output: <span className="text-emerald-400">768</span></div>
                </div>
            </div>

            {/* Causal Masking */}
            <div className="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-yellow-400">⚠️ Causal Masking</h3>
                <p className="text-gray-300">
                    GPT-2 is a <strong>decoder-only</strong> model. During training, we prevent it from looking at future tokens by applying a <strong>causal mask</strong> (upper triangle = -∞ before softmax).
                </p>
                <p className="text-gray-300 text-sm">
                    This ensures the model learns to predict the next token using only past context, mimicking real autoregressive generation.
                </p>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-400">📝 Exercise</h3>
                <p className="text-gray-300">
                    Why does GPT-2 use causal masking? What would happen without it?
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
                <div className="text-gray-400 flex items-center">
                    Steps 4-9 coming soon!
                </div>
            </div>
        </div>
    );
}
