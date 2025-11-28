import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

// Example Data
// Prediction (Softmax output): [0.1, 0.7, 0.2]
// True Label (One-Hot): [0, 1, 0]
// Correct Class Index: 1
// Loss: -ln(0.7) ≈ 0.36

const predictions = [0.1, 0.7, 0.2];
const labels = [0, 1, 0];
const correctIndex = 1;
const lossValue = -Math.log(predictions[correctIndex]);

const COLORS = {
    pred: 0x5b9bd5,       // Blue
    label: 0x70ad47,      // Green
    highlight: 0xffc000,  // Yellow/Gold
    loss: 0xed7d31,       // Orange
    bg: 0xffffff,
    text: '#333333'
};

const STEPS = [
    {
        id: 'show-vectors',
        title: 'Model Output vs True Label',
        desc: 'The model predicts probabilities (Blue). The true label is One-Hot encoded (Green).'
    },
    {
        id: 'filter',
        title: 'Select Correct Class',
        desc: 'Cross-Entropy only cares about the probability of the CORRECT class (where Label=1).'
    },
    {
        id: 'log',
        title: 'Logarithm',
        desc: 'Take the natural log of the predicted probability: ln(0.7) ≈ -0.36'
    },
    {
        id: 'negate',
        title: 'Negate (Loss)',
        desc: 'Negate the result to get a positive Loss value: -(-0.36) = 0.36'
    },
];

export default function CrossEntropyPanel({ onStepChange }) {
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const objectsRef = useRef({});
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [explanation, setExplanation] = useState('Click Play to see Cross-Entropy calculation');

    useEffect(() => {
        if (onStepChange) {
            onStepChange(step);
        }
    }, [step, onStepChange]);

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

        const cellSize = 60;
        const gap = 10;

        const createCell = (value, x, y, color, labelText, visible = true) => {
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
            canvas.width = 256;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.font = 'bold 48px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(typeof value === 'number' ? value.toFixed(1) : value, 128, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const valMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const valSprite = new THREE.Sprite(valMaterial);
            valSprite.scale.set(cellSize * 1.5, cellSize * 0.75, 1);
            group.add(valSprite);

            if (labelText) {
                const lblCanvas = document.createElement('canvas');
                lblCanvas.width = 256;
                lblCanvas.height = 64;
                const lblCtx = lblCanvas.getContext('2d');
                lblCtx.fillStyle = '#333';
                lblCtx.font = 'bold 32px Arial';
                lblCtx.textAlign = 'center';
                lblCtx.textBaseline = 'middle';
                lblCtx.fillText(labelText, 128, 32);

                const lblTexture = new THREE.CanvasTexture(lblCanvas);
                const lblMaterial = new THREE.SpriteMaterial({ map: lblTexture, transparent: true });
                const lblSprite = new THREE.Sprite(lblMaterial);
                lblSprite.scale.set(cellSize * 1.5, cellSize * 0.4, 1);
                lblSprite.position.y = cellSize * 0.8;
                group.add(lblSprite);
            }

            group.position.set(x, y, 0);
            group.visible = visible;
            group.userData = { value, mesh, material };
            scene.add(group);
            return group;
        };

        const createText = (text, x, y, size = 40, color = '#333') => {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = color;
            ctx.font = `bold ${size}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 256, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.scale.set(200, 50, 1);
            sprite.position.set(x, y, 0);
            sprite.visible = false;
            scene.add(sprite);
            return sprite;
        };

        // Layout
        const startX = -150;
        const startY = 80;

        // 1. Prediction Vector (Vertical)
        const predCells = predictions.map((val, i) =>
            createCell(val, startX, startY - i * (cellSize + gap), COLORS.pred, i === 0 ? 'Pred (p)' : '', false)
        );

        // 2. Label Vector (Vertical)
        const labelCells = labels.map((val, i) =>
            createCell(val, startX + 120, startY - i * (cellSize + gap), COLORS.label, i === 0 ? 'Label (y)' : '', false)
        );

        // 3. Calculation Area
        const calcX = 150;
        const calcY = 0;

        const pVal = createCell(0.7, calcX - 80, calcY, COLORS.pred, 'p', false);
        const logVal = createCell(-0.36, calcX + 20, calcY, COLORS.loss, 'ln(p)', false);
        const negLogVal = createCell(0.36, calcX + 120, calcY, COLORS.loss, '-ln(p)', false);

        const arrow1 = createText('→', calcX - 30, calcY);
        const arrow2 = createText('→', calcX + 70, calcY);
        const formulaText = createText('Loss = -ln(p)', calcX + 20, calcY + 80, 36);

        objectsRef.current = {
            predCells, labelCells,
            pVal, logVal, negLogVal,
            arrow1, arrow2, formulaText
        };

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
            case 1: // Show Vectors
                objs.predCells.forEach((cell, i) => {
                    cell.visible = true;
                    cell.scale.set(0.01, 0.01, 0.01);
                    gsap.to(cell.scale, { x: 1, y: 1, z: 1, duration: 0.5, delay: i * 0.1, ease: 'back.out' });
                });
                objs.labelCells.forEach((cell, i) => {
                    cell.visible = true;
                    cell.scale.set(0.01, 0.01, 0.01);
                    gsap.to(cell.scale, { x: 1, y: 1, z: 1, duration: 0.5, delay: 0.3 + i * 0.1, ease: 'back.out' });
                });
                break;
            case 2: // Filter (Select Correct Class)
                // Dim incorrect classes
                objs.predCells.forEach((cell, i) => {
                    if (i !== correctIndex) {
                        gsap.to(cell.userData.material, { opacity: 0.2, duration: 0.5 });
                    } else {
                        gsap.to(cell.scale, { x: 1.2, y: 1.2, duration: 0.3, yoyo: true, repeat: 1 });
                    }
                });
                objs.labelCells.forEach((cell, i) => {
                    if (i !== correctIndex) {
                        gsap.to(cell.userData.material, { opacity: 0.2, duration: 0.5 });
                    }
                });

                // Move selected p to calculation area
                objs.pVal.visible = true;
                objs.pVal.position.set(objs.predCells[correctIndex].position.x, objs.predCells[correctIndex].position.y, 0);
                gsap.to(objs.pVal.position, { x: 70, y: 0, duration: 1, ease: 'power2.inOut' });
                break;
            case 3: // Log
                objs.arrow1.visible = true;
                objs.logVal.visible = true;
                objs.logVal.scale.set(0.01, 0.01, 0.01);
                gsap.to(objs.logVal.scale, { x: 1, y: 1, z: 1, duration: 0.5, ease: 'back.out' });
                break;
            case 4: // Negate
                objs.arrow2.visible = true;
                objs.negLogVal.visible = true;
                objs.negLogVal.scale.set(0.01, 0.01, 0.01);
                gsap.to(objs.negLogVal.scale, { x: 1, y: 1, z: 1, duration: 0.5, ease: 'back.out' });
                objs.formulaText.visible = true;
                break;
        }
    };

    const playAnimation = async () => {
        if (isPlaying) return;
        setIsPlaying(true);
        reset();

        for (let i = 0; i < STEPS.length; i++) {
            setStep(i + 1);
            setExplanation(`${STEPS[i].title}\n${STEPS[i].desc}`);
            animateStep(i + 1);
            await new Promise(r => setTimeout(r, 2000));
        }

        setExplanation('Complete! High probability for correct class = Low Loss.');
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

        // Hide all
        Object.values(objs).flat().forEach(obj => {
            if (obj) obj.visible = false;
        });

        // Reset opacities
        objs.predCells.forEach(cell => cell.userData.material.opacity = 0.8);
        objs.labelCells.forEach(cell => cell.userData.material.opacity = 0.8);

        setStep(0);
        setExplanation('Click Play to see Cross-Entropy calculation');
    };

    return (
        <div className="flex flex-col items-center p-3">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Cross-Entropy Animation</h2>

            <div ref={containerRef} className="w-full rounded-lg overflow-hidden shadow-lg bg-white" />

            <div className="mt-2 p-2 bg-white rounded-lg w-full text-center shadow h-20 flex items-center justify-center">
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
        </div>
    );
}
