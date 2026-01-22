import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import gsap from 'gsap';

const COLORS = {
    input: 0x5b9bd5,      // Blue
    window: 0xffc000,     // Yellow
    maxHighlight: 0xef4444, // Red
    output: 0x70ad47,     // Green
    bg: 0xffffff
};

export default function MaxPoolingPanel({ inputMatrix, poolSize, stride, onComplete }) {
    const containerRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const objectsRef = useRef({});
    const [isRunning, setIsRunning] = useState(false);
    const [currentPos, setCurrentPos] = useState({ row: 0, col: 0 });

    useEffect(() => {
        if (!containerRef.current) return;

        const width = containerRef.current.clientWidth;
        const height = 500;

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

        const cellSize = 50;
        const gap = 5;

        const createCell = (value, x, y, color) => {
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
            ctx.fillStyle = 'white';
            ctx.font = 'bold 64px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(value.toString(), 64, 64);

            const texture = new THREE.CanvasTexture(canvas);
            const valMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const valSprite = new THREE.Sprite(valMaterial);
            valSprite.scale.set(cellSize * 0.8, cellSize * 0.8, 1);
            group.add(valSprite);

            group.position.set(x, y, 0);
            group.userData = { value, mesh, material };
            scene.add(group);
            return group;
        };

        // Input matrix (left side)
        const inputSize = inputMatrix.length;
        const inputCells = [];
        const startX = -200;
        const startY = 100;

        inputMatrix.forEach((row, i) => {
            const rowCells = [];
            row.forEach((val, j) => {
                const x = startX + j * (cellSize + gap);
                const y = startY - i * (cellSize + gap);
                const cell = createCell(val, x, y, COLORS.input);
                rowCells.push(cell);
            });
            inputCells.push(rowCells);
        });

        // Window outline
        const windowGeom = new THREE.PlaneGeometry(
            poolSize * cellSize + (poolSize - 1) * gap,
            poolSize * cellSize + (poolSize - 1) * gap
        );
        const windowMaterial = new THREE.MeshBasicMaterial({
            color: COLORS.window,
            transparent: true,
            opacity: 0.3
        });
        const window = new THREE.Mesh(windowGeom, windowMaterial);
        const windowBorder = new THREE.LineSegments(
            new THREE.EdgesGeometry(windowGeom),
            new THREE.LineBasicMaterial({ color: COLORS.window, linewidth: 4 })
        );
        window.add(windowBorder);
        window.visible = false;
        scene.add(window);

        // Output matrix (right side)
        const outputSize = Math.floor((inputSize - poolSize) / stride) + 1;
        const outputCells = [];
        const outputStartX = 100;
        const outputStartY = 100;

        for (let i = 0; i < outputSize; i++) {
            const rowCells = [];
            for (let j = 0; j < outputSize; j++) {
                const x = outputStartX + j * (cellSize + gap);
                const y = outputStartY - i * (cellSize + gap);
                const cell = createCell(0, x, y, COLORS.output);
                cell.visible = false;
                rowCells.push(cell);
            }
            outputCells.push(rowCells);
        }

        objectsRef.current = { inputCells, outputCells, window, cellSize, gap, startX, startY };

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
    }, [inputMatrix, poolSize, stride]);

    const runPooling = async () => {
        if (isRunning) return;
        setIsRunning(true);

        const { inputCells, outputCells, window, cellSize, gap, startX, startY } = objectsRef.current;
        const inputSize = inputMatrix.length;
        const outputSize = Math.floor((inputSize - poolSize) / stride) + 1;

        window.visible = true;

        let outRow = 0;
        let outCol = 0;

        for (let i = 0; i <= inputSize - poolSize; i += stride) {
            for (let j = 0; j <= inputSize - poolSize; j += stride) {
                setCurrentPos({ row: i, col: j });

                // Position window
                const windowX = startX + j * (cellSize + gap) + (poolSize * cellSize + (poolSize - 1) * gap) / 2 - cellSize / 2;
                const windowY = startY - i * (cellSize + gap) - (poolSize * cellSize + (poolSize - 1) * gap) / 2 + cellSize / 2;

                await new Promise(resolve => {
                    gsap.to(window.position, {
                        x: windowX,
                        y: windowY,
                        duration: 0.3,
                        ease: 'power2.inOut',
                        onComplete: resolve
                    });
                });

                // Find max in window
                let maxVal = -Infinity;
                let maxR = -1, maxC = -1;
                for (let r = i; r < i + poolSize; r++) {
                    for (let c = j; c < j + poolSize; c++) {
                        if (inputMatrix[r][c] > maxVal) {
                            maxVal = inputMatrix[r][c];
                            maxR = r;
                            maxC = c;
                        }
                    }
                }

                // Highlight max cell
                const maxCell = inputCells[maxR][maxC];
                const originalColor = maxCell.userData.material.color.getHex();

                await new Promise(resolve => {
                    gsap.to(maxCell.userData.material.color, {
                        r: (COLORS.maxHighlight >> 16 & 255) / 255,
                        g: (COLORS.maxHighlight >> 8 & 255) / 255,
                        b: (COLORS.maxHighlight & 255) / 255,
                        duration: 0.3,
                        yoyo: true,
                        repeat: 1,
                        onComplete: resolve
                    });
                });

                // Show output cell with max value
                const outputCell = outputCells[outRow][outCol];
                outputCell.visible = true;

                // Update output cell value
                const valSprite = outputCell.children[2];
                const canvas = document.createElement('canvas');
                canvas.width = 128;
                canvas.height = 128;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'white';
                ctx.font = 'bold 64px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(maxVal.toString(), 64, 64);
                valSprite.material.map = new THREE.CanvasTexture(canvas);

                await new Promise(r => setTimeout(r, 300));

                outCol++;
                if (outCol >= outputSize) {
                    outCol = 0;
                    outRow++;
                }
            }
        }

        window.visible = false;
        setIsRunning(false);

        if (onComplete) {
            onComplete();
        }
    };

    const reset = () => {
        if (isRunning) return;
        const { outputCells, window } = objectsRef.current;

        outputCells.forEach(row => {
            row.forEach(cell => {
                cell.visible = false;
            });
        });

        window.visible = false;
        setCurrentPos({ row: 0, col: 0 });
    };

    return (
        <div className="flex flex-col items-center p-3">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Max Pooling Animation</h2>

            <div ref={containerRef} className="w-full rounded-lg overflow-hidden shadow-lg bg-white" />

            <div className="mt-2 p-2 bg-white rounded-lg w-full text-center shadow">
                <p className="text-sm text-gray-800">
                    {isRunning ?
                        `Processing window at position (${currentPos.row}, ${currentPos.col})` :
                        'Click Run to start pooling'}
                </p>
            </div>

            <div className="flex gap-2 mt-2">
                <button
                    onClick={runPooling}
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
