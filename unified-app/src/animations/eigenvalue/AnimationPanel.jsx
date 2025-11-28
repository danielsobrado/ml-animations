import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

//Symmetric matrix example: A = [[3, 1], [1, 2]]
// Eigenvalues: λ₁ ≈ 3.62, λ₂ ≈ 1.38
// Eigenvectors normalized
const matrixA = [[3, 1], [1, 2]];
const lambda1 = 3.62;
const lambda2 = 1.38;
const eigenvector1 = [0.85, 0.53]; // Normalized
const eigenvector2 = [-0.53, 0.85]; // Normalized

const matrixQ = [
    [0.85, -0.53],
    [0.53, 0.85]
];

const matrixLambda = [[3.62, 0], [0, 1.38]];

const COLORS = {
    matrixA: 0xed7d31,
    matrixQ: 0x5b9bd5,
    matrixLambda: 0x70ad47,
    eigenvec: 0xffc000,
    text: 0x333333,
    bg: 0xffffff
};

const STEPS = [
    { id: 'show-a', title: 'Symmetric Matrix A', desc: 'A is a 2×2 symmetric matrix (A = A^T)' },
    { id: 'eigen-concept', title: 'Eigenvalue Equation', desc: 'A v = λ v (eigenvector v, eigenvalue λ)' },
    { id: 'show-eigen1', title: 'First Eigenvector', desc: 'v₁ direction with eigenvalue λ₁ = 3.62' },
    { id: 'show-eigen2', title: 'Second Eigenvector', desc: 'v₂ direction with eigenvalue λ₂ = 1.38 (perpendicular to v₁)' },
    { id: 'show-q', title: 'Matrix Q', desc: 'Q = [v₁ v₂]: eigenvectors as columns' },
    { id: 'show-lambda', title: 'Matrix Λ', desc: 'Λ diagonal: eigenvalues on diagonal' },
    { id: 'decomposition', title: 'Eigenvalue Decomposition', desc: 'A = Q Λ Q^T (for symmetric A)' },
];

export default function AnimationPanel() {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [explanation, setExplanation] = useState('Click Play to see eigenvalue decomposition');
    const objectsRef = useRef({ cellsA: [], cellsQ: [], cellsLambda: [], arrows: [], labels: [] });

    useEffect(() => {
        if (!containerRef.current) return;

        objectsRef.current = { cellsA: [], cellsQ: [], cellsLambda: [], arrows: [], labels: [] };

        const width = containerRef.current.clientWidth;
        const height = 450;

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

        const cellSize = 40;
        const gap = 4;

        const createCell = (value, x, y, color, visible = true) => {
            const group = new THREE.Group();

            const geometry = new THREE.PlaneGeometry(cellSize, cellSize);
            const material = new THREE.MeshBasicMaterial({
                color,
                transparent: true,
                opacity: 0.8
            });
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);

            const border = new THREE.LineSegments(
                new THREE.EdgesGeometry(geometry),
                new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 2 })
            );
            group.add(border);

            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.font = value.toString().length > 4 ? 'bold 42px Arial' : 'bold 50px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toString(), 64, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const labelMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const label = new THREE.Sprite(labelMaterial);
            label.scale.set(30, 30, 1);
            group.add(label);

            group.position.set(x, y, 0);
            group.visible = visible;
            group.userData = { value, originalColor: color, mesh, label, canvas, texture };
            scene.add(group);
            return group;
        };

        const createLabel = (text, x, y, size = 32) => {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.font = `bold ${size * 2}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 256, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.scale.set(text.length * size / 2, size, 1);
            sprite.position.set(x, y, 0);
            sprite.visible = false;
            scene.add(sprite);
            return sprite;
        };

        // Matrix A (2x2) at top
        const aX = -30;
        const aY = 140;
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                const cell = createCell(
                    matrixA[i][j],
                    aX + j * (cellSize + gap),
                    aY - i * (cellSize + gap),
                    COLORS.matrixA,
                    false
                );
                objectsRef.current.cellsA.push(cell);
            }
        }
        const labelA = createLabel('A', aX + 22, aY + 50);
        objectsRef.current.labels.push(labelA);

        // Matrix Q (2x2)
        const qX = -180;
        const qY = 30;
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                const val = matrixQ[i][j].toFixed(2);
                const cell = createCell(
                    val,
                    qX + j * (cellSize + gap),
                    qY - i * (cellSize + gap),
                    COLORS.matrixQ,
                    false
                );
                objectsRef.current.cellsQ.push(cell);
            }
        }
        const labelQ = createLabel('Q', qX + 22, qY + 60);
        objectsRef.current.labels.push(labelQ);

        // Matrix Lambda (2x2)
        const lambdaX = -50;
        const lambdaY = 30;
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                const val = matrixLambda[i][j].toFixed(2);
                const cell = createCell(
                    val,
                    lambdaX + j * (cellSize + gap),
                    lambdaY - i * (cellSize + gap),
                    COLORS.matrixLambda,
                    false
                );
                objectsRef.current.cellsLambda.push(cell);
            }
        }
        const labelLambda = createLabel('Λ', lambdaX + 22, lambdaY + 60);
        objectsRef.current.labels.push(labelLambda);

        const multLabel = createLabel('×', qX + 100, qY, 28);
        const equalsLabel = createLabel('=', aX - 40, aY - 20, 32);
        objectsRef.current.labels.push(multLabel, equalsLabel);

        rendererRef.current = renderer;

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

    const animateStep = (stepIndex) => {
        const objs = objectsRef.current;

        switch (stepIndex) {
            case 1: // Show A
                objs.cellsA.forEach((cell, i) => {
                    cell.visible = true;
                    cell.scale.set(0.01, 0.01, 0.01);
                    gsap.to(cell.scale, { x: 1, y: 1, z: 1, duration: 0.4, delay: i * 0.1, ease: 'back.out' });
                });
                objs.labels[0].visible = true;
                break;
            case 2: // Concept
                // Just explanation
                break;
            case 3: // First eigenvector
                objs.cellsA.forEach((cell, i) => {
                    if (i % 2 === 0) { // Highlight first column pattern
                        gsap.to(cell.scale, { x: 1.15, y: 1.15, duration: 0.3 });
                    }
                });
                setTimeout(() => {
                    objs.cellsA.forEach(cell => {
                        gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
                    });
                }, 1000);
                break;
            case 4: // Second eigenvector
                objs.cellsA.forEach((cell, i) => {
                    if (i % 2 === 1) { // Highlight second column pattern
                        gsap.to(cell.scale, { x: 1.15, y: 1.15, duration: 0.3 });
                    }
                });
                setTimeout(() => {
                    objs.cellsA.forEach(cell => {
                        gsap.to(cell.scale, { x: 1, y: 1, duration: 0.3 });
                    });
                }, 1000);
                break;
            case 5: // Show Q
                objs.labels[1].visible = true;
                objs.cellsQ.forEach((cell, i) => {
                    cell.visible = true;
                    cell.scale.set(0.01, 0.01, 0.01);
                    gsap.to(cell.scale, { x: 1, y: 1, z: 1, duration: 0.4, delay: i * 0.08, ease: 'back.out' });
                });
                objs.labels[3].visible = true; // mult
                break;
            case 6: // Show Lambda
                objs.labels[2].visible = true;
                objs.cellsLambda.forEach((cell, i) => {
                    cell.visible = true;
                    cell.scale.set(0.01, 0.01, 0.01);
                    gsap.to(cell.scale, { x: 1, y: 1, z: 1, duration: 0.4, delay: i * 0.1, ease: 'back.out' });
                });
                objs.labels[4].visible = true; // equals
                break;
            case 7: // Decomposition
                [...objs.cellsQ, ...objs.cellsLambda].forEach(cell => {
                    gsap.to(cell.userData.mesh.material, {
                        opacity: 0.3, duration: 0.2, yoyo: true, repeat: 2
                    });
                });
                objs.cellsA.forEach(cell => {
                    gsap.to(cell.userData.mesh.material, {
                        opacity: 0.5, duration: 0.2, yoyo: true, repeat: 3
                    });
                });
                break;
        }
    };

    const playAnimation = async () => {
        if (isPlaying) return;
        setIsPlaying(true);

        for (let i = step; i < STEPS.length; i++) {
            setStep(i + 1);
            setExplanation(`${STEPS[i].title}\n${STEPS[i].desc}`);
            animateStep(i + 1);
            await new Promise(r => setTimeout(r, 1600));
        }

        setExplanation('Complete! Eigenvalue decomposition: A = Q Λ Q^T for symmetric matrices.');
        setIsPlaying(false);
    };

    const nextStep = () => {
        if (isPlaying || step >= STEPS.length) return;
        const newStep = step + 1;
        setStep(newStep);
        setExplanation(`${STEPS[newStep - 1].title}\n${STEPS[newStep - 1].desc}`);
        animateStep(newStep);
    };

    const prevStep = () => {
        if (isPlaying || step <= 0) return;
        reset();
        const targetStep = step - 1;
        setTimeout(() => {
            for (let i = 1; i <= targetStep; i++) {
                animateStep(i);
            }
            setStep(targetStep);
            if (targetStep > 0) {
                setExplanation(`${STEPS[targetStep - 1].title}\n${STEPS[targetStep - 1].desc}`);
            }
        }, 100);
    };

    const reset = () => {
        if (isPlaying) return;
        const objs = objectsRef.current;

        [...objs.cellsA, ...objs.cellsQ, ...objs.cellsLambda].forEach(cell => {
            if (cell) {
                cell.visible = false;
                cell.scale.set(1, 1, 1);
                if (cell.userData.mesh) {
                    cell.userData.mesh.material.opacity = 0.8;
                }
            }
        });

        objs.labels.forEach(label => {
            if (label) label.visible = false;
        });

        setStep(0);
        setExplanation('Click Play to see eigenvalue decomposition');
    };

    return (
        <div className="flex flex-col items-center p-3">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Animation Demo</h2>

            <div ref={containerRef} className="w-full rounded-lg overflow-hidden shadow-lg bg-white" />

            <div className="mt-2 p-2 bg-white rounded-lg w-full text-center shadow">
                <p className="text-gray-800 whitespace-pre-line text-sm">{explanation}</p>
            </div>

            <div className="flex items-center gap-2 mt-2">
                <button
                    onClick={prevStep}
                    disabled={isPlaying || step <= 0}
                    className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    ← Prev
                </button>

                <div className="px-3 py-1 bg-gray-200 rounded-lg font-mono text-gray-700 min-w-[80px] text-center text-sm">
                    {step} / {STEPS.length}
                </div>

                <button
                    onClick={nextStep}
                    disabled={isPlaying || step >= STEPS.length}
                    className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    Next →
                </button>
            </div>

            <div className="flex gap-2 mt-2">
                <button
                    onClick={playAnimation}
                    disabled={isPlaying || step >= STEPS.length}
                    className="px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    {isPlaying ? 'Playing...' : '▶ Play'}
                </button>
                <button
                    onClick={reset}
                    disabled={isPlaying}
                    className="px-4 py-2 bg-red-500 hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors text-sm"
                >
                    ↺ Reset
                </button>
            </div>

            <div className="mt-2 p-2 bg-green-100 rounded-lg w-full text-center border border-green-300">
                <p className="text-green-800 text-xs font-medium">
                    A v = λ v | A = Q Λ Q^T | Q^T Q = I (orthogonal eigenvectors)
                </p>
            </div>
        </div>
    );
}
