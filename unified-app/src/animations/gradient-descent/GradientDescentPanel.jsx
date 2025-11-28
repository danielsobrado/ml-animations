import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

// Loss function: L(w) = w^2
// Gradient: dL/dw = 2w
// Update: w_new = w_old - alpha * 2w_old

const COLORS = {
    curve: 0x7030a0,      // Purple
    ball: 0x5b9bd5,       // Blue
    gradient: 0xed7d31,   // Orange
    optimal: 0x70ad47,    // Green
    bg: 0xffffff
};

export default function GradientDescentPanel({ learningRate, startWeight, onStepChange }) {
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const objectsRef = useRef({});
    const [isRunning, setIsRunning] = useState(false);
    const [currentWeight, setCurrentWeight] = useState(startWeight);
    const [iteration, setIteration] = useState(0);

    useEffect(() => {
        setCurrentWeight(startWeight);
        setIteration(0);
    }, [startWeight]);

    useEffect(() => {
        if (onStepChange) {
            const loss = currentWeight * currentWeight;
            onStepChange(iteration, currentWeight, loss);
        }
    }, [iteration, currentWeight, onStepChange]);

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 400;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(COLORS.bg);
        sceneRef.current = scene;

        const camera = new THREE.OrthographicCamera(
            width / -2, width / 2, height / 2, height / -2, 0.1, 1000
        );
        camera.position.z = 100;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Draw loss curve using line segments
        const curvePoints = [];
        const wRange = 5; // Display w from -5 to 5
        const scale = 50; // Scale for display

        for (let w = -wRange; w <= wRange; w += 0.1) {
            const loss = w * w;
            const x = w * scale;
            const y = -loss * 10 + 100; // Invert y and offset
            curvePoints.push(new THREE.Vector3(x, y, 0));
        }

        const curveGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const curveMaterial = new THREE.LineBasicMaterial({ color: COLORS.curve, linewidth: 3 });
        const curve = new THREE.Line(curveGeometry, curveMaterial);
        scene.add(curve);

        // Ball
        const ballGeometry = new THREE.CircleGeometry(8, 32);
        const ballMaterial = new THREE.MeshBasicMaterial({ color: COLORS.ball });
        const ball = new THREE.Mesh(ballGeometry, ballMaterial);
        scene.add(ball);

        // Gradient arrow (will be created dynamically)
        objectsRef.current = { ball, scene, scale };

        // Position ball at start
        updateBallPosition(startWeight);

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

    const updateBallPosition = (w) => {
        const { ball, scale } = objectsRef.current;
        const loss = w * w;
        const x = w * scale;
        const y = -loss * 10 + 100;
        ball.position.set(x, y, 0);
    };

    const runGradientDescent = async () => {
        if (isRunning) return;
        setIsRunning(true);

        let w = startWeight;
        setCurrentWeight(w);
        setIteration(0);
        updateBallPosition(w);

        const maxIterations = 50;
        const convergenceThreshold = 0.01;

        for (let i = 0; i < maxIterations; i++) {
            // Calculate gradient
            const gradient = 2 * w;

            // Update weight
            const wNew = w - learningRate * gradient;

            // Animate ball movement
            const { ball, scale } = objectsRef.current;
            const xNew = wNew * scale;
            const yNew = -(wNew * wNew) * 10 + 100;

            await new Promise(resolve => {
                gsap.to(ball.position, {
                    x: xNew,
                    y: yNew,
                    duration: 0.5,
                    ease: 'power2.inOut',
                    onComplete: resolve
                });
            });

            w = wNew;
            setCurrentWeight(w);
            setIteration(i + 1);

            // Check convergence
            if (Math.abs(w) < convergenceThreshold) {
                break;
            }

            // Check divergence
            if (Math.abs(w) > 10) {
                break;
            }

            await new Promise(r => setTimeout(r, 200));
        }

        setIsRunning(false);
    };

    const reset = () => {
        if (isRunning) return;
        setCurrentWeight(startWeight);
        setIteration(0);
        updateBallPosition(startWeight);
    };

    return (
        <div className="flex flex-col items-center p-3">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Loss Landscape</h2>

            <div ref={containerRef} className="w-full rounded-lg overflow-hidden shadow-lg bg-white" />

            <div className="mt-2 p-2 bg-white rounded-lg w-full text-center shadow">
                <p className="text-sm text-gray-800">
                    Iteration: <span className="font-bold">{iteration}</span> |
                    Weight: <span className="font-bold text-blue-600">{currentWeight.toFixed(3)}</span> |
                    Loss: <span className="font-bold text-purple-600">{(currentWeight * currentWeight).toFixed(3)}</span>
                </p>
            </div>

            <div className="flex gap-2 mt-2">
                <button
                    onClick={runGradientDescent}
                    disabled={isRunning}
                    className="px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    {isRunning ? 'Running...' : '▶ Run'}
                </button>
                <button
                    onClick={reset}
                    disabled={isRunning}
                    className="px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    ↺ Reset
                </button>
            </div>
        </div>
    );
}
