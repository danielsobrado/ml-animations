import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

export default function Step4FFN({ onComplete, onNext, onPrev }) {
    const containerRef = useRef(null);
    const [quizAnswer, setQuizAnswer] = useState('');
    const [quizFeedback, setQuizFeedback] = useState('');

    // Visualization of expansion and contraction
    useEffect(() => {
        if (!containerRef.current) return;

        const width = 600;
        const height = 300;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1f2937);

        const camera = new THREE.OrthographicCamera(
            width / -2, width / 2, height / 2, height / -2, 0.1, 1000
        );
        camera.position.z = 100;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        // Create nodes
        const createLayer = (count, x, color, label) => {
            const group = new THREE.Group();
            const spacing = 25;
            const startY = (count * spacing) / 2 - spacing / 2;

            for (let i = 0; i < count; i++) {
                const geometry = new THREE.CircleGeometry(6, 32);
                const material = new THREE.MeshBasicMaterial({ color });
                const circle = new THREE.Mesh(geometry, material);
                circle.position.y = startY - i * spacing;
                group.add(circle);
            }

            group.position.x = x;
            scene.add(group);
            return group;
        };

        // Input Layer (d_model)
        const inputLayer = createLayer(8, -200, 0x5b9bd5, 'Input');

        // Hidden Layer (4 * d_model)
        const hiddenLayer = createLayer(32, 0, 0xffc000, 'Hidden');

        // Output Layer (d_model)
        const outputLayer = createLayer(8, 200, 0x70ad47, 'Output');

        // Connections (animated)
        const linesMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.1
        });

        const drawConnections = (layer1, layer2) => {
            layer1.children.forEach(n1 => {
                layer2.children.forEach(n2 => {
                    const points = [n1.position.clone().add(layer1.position), n2.position.clone().add(layer2.position)];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(geometry, linesMaterial);
                    scene.add(line);
                });
            });
        };

        drawConnections(inputLayer, hiddenLayer);
        drawConnections(hiddenLayer, outputLayer);

        // Animation loop
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
    }, []);

    const checkQuiz = () => {
        const correct = quizAnswer.toLowerCase().includes('4') || quizAnswer.toLowerCase().includes('four');
        setQuizFeedback(correct
            ? '✓ Correct! The internal dimension is typically 4 times the embedding dimension.'
            : '✗ Try again. How much larger is the hidden layer compared to the input?'
        );
        if (correct) onComplete();
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold mb-2">Step 4: Feed-Forward Network</h2>
                <p className="text-gray-800 dark:text-gray-400">Processing information position-wise</p>
            </div>

            {/* Explanation */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">The "Brain" of the Block</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    After attention gathers information from other tokens, the <strong>Feed-Forward Network (FFN)</strong> processes each token <em>independently</em>.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                    It consists of two linear transformations with a non-linear activation function in between.
                </p>
            </div>

            {/* Visualization */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">Expansion & Contraction</h3>
                <div ref={containerRef} className="border border-gray-700 rounded flex justify-center overflow-hidden" />
                <div className="flex justify-between text-sm text-gray-800 dark:text-gray-400 px-10">
                    <span>Input (d_model)</span>
                    <span>Hidden (4 × d_model)</span>
                    <span>Output (d_model)</span>
                </div>
            </div>

            {/* Formula */}
            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">The Math</h3>
                <div className="bg-gray-900 p-4 rounded space-y-2 font-mono text-sm">
                    <div className="text-gray-700 dark:text-gray-300">FFN(x) = GELU(xW<sub>1</sub> + b<sub>1</sub>)W<sub>2</sub> + b<sub>2</sub></div>
                </div>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 ml-4">
                    <li><strong>Expansion</strong>: Project from 768 to 3072 dimensions</li>
                    <li><strong>Activation</strong>: Apply GELU (Gaussian Error Linear Unit)</li>
                    <li><strong>Contraction</strong>: Project back to 768 dimensions</li>
                </ul>
            </div>

            {/* Exercise */}
            <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-6 space-y-4">
                <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">📝 Exercise</h3>
                <p className="text-gray-700 dark:text-gray-300">
                    If the embedding dimension (d_model) is 768, what is the dimension of the hidden layer in the FFN?
                </p>
                <input
                    type="text"
                    value={quizAnswer}
                    onChange={(e) => setQuizAnswer(e.target.value)}
                    className="w-full bg-gray-700 text-white px-4 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    placeholder="Enter a number..."
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
                    Next: Layer Norm & Residuals →
                </button>
            </div>
        </div>
    );
}
