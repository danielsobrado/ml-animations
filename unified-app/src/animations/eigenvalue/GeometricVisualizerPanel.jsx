import React, { useRef, useEffect, useState } from 'react';

// Matrix for transformation: [[3, 1], [1, 2]]
// Eigenvalues: λ₁ ≈ 3.62, λ₂ ≈ 1.38
// Eigenvectors: v₁ ≈ [0.85, 0.53], v₂ ≈ [-0.53, 0.85]
const MATRIX_A = [[3, 1], [1, 2]];
const EIGENVALUES = [3.62, 1.38];
const EIGENVECTORS = [
    [0.85, 0.53],   // v1
    [-0.53, 0.85]   // v2
];

export default function GeometricVisualizerPanel() {
    const canvasRef = useRef(null);
    const [showTransformed, setShowTransformed] = useState(false);
    const [showEigenvectors, setShowEigenvectors] = useState(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 60;

        const draw = () => {
            // Clear canvas
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);

            // Draw grid
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            for (let i = -5; i <= 5; i++) {
                // Vertical lines
                ctx.beginPath();
                ctx.moveTo(centerX + i * scale, 0);
                ctx.lineTo(centerX + i * scale, height);
                ctx.stroke();
                // Horizontal lines
                ctx.beginPath();
                ctx.moveTo(0, centerY + i * scale);
                ctx.lineTo(width, centerY + i * scale);
                ctx.stroke();
            }

            // Draw axes
            ctx.strokeStyle = '#999999';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(centerX, 0);
            ctx.lineTo(centerX, height);
            ctx.stroke();

            // Draw unit circle (or ellipse if transformed)
            if (!showTransformed) {
                // Original unit circle
                ctx.strokeStyle = '#5b9bd5';
                ctx.lineWidth = 3;
                ctx.beginPath();
                for (let angle = 0; angle <= Math.PI * 2; angle += 0.01) {
                    const x = Math.cos(angle);
                    const y = Math.sin(angle);
                    const screenX = centerX + x * scale;
                    const screenY = centerY - y * scale;
                    if (angle === 0) {
                        ctx.moveTo(screenX, screenY);
                    } else {
                        ctx.lineTo(screenX, screenY);
                    }
                }
                ctx.stroke();

                // Label
                ctx.fillStyle = '#5b9bd5';
                ctx.font = 'bold 16px Arial';
                ctx.fillText('Unit Circle', centerX + 70, centerY - 70);
            } else {
                // Transformed ellipse (Av for unit circle points)
                ctx.strokeStyle = '#ed7d31';
                ctx.lineWidth = 3;
                ctx.beginPath();
                for (let angle = 0; angle <= Math.PI * 2; angle += 0.01) {
                    const x = Math.cos(angle);
                    const y = Math.sin(angle);

                    // Apply matrix transformation: [x', y'] = A * [x, y]
                    const xPrime = MATRIX_A[0][0] * x + MATRIX_A[0][1] * y;
                    const yPrime = MATRIX_A[1][0] * x + MATRIX_A[1][1] * y;

                    const screenX = centerX + xPrime * scale;
                    const screenY = centerY - yPrime * scale;

                    if (angle === 0) {
                        ctx.moveTo(screenX, screenY);
                    } else {
                        ctx.lineTo(screenX, screenY);
                    }
                }
                ctx.stroke();

                // Label
                ctx.fillStyle = '#ed7d31';
                ctx.font = 'bold 16px Arial';
                ctx.fillText('Transformed (Ellipse)', centerX + 90, centerY - 100);
            }

            // Draw eigenvectors if enabled
            if (showEigenvectors) {
                // Eigenvector 1 (longer)
                const v1x = EIGENVECTORS[0][0];
                const v1y = EIGENVECTORS[0][1];
                const length1 = showTransformed ? EIGENVALUES[0] : 1;

                ctx.strokeStyle = '#70ad47';
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(centerX + v1x * length1 * scale, centerY - v1y * length1 * scale);
                ctx.stroke();

                // Arrow head
                drawArrowHead(ctx, centerX, centerY, centerX + v1x * length1 * scale, centerY - v1y * length1 * scale, '#70ad47');

                // Label
                ctx.fillStyle = '#70ad47';
                ctx.font = 'bold 14px Arial';
                ctx.fillText(showTransformed ? `λ₁v₁ (λ₁=${EIGENVALUES[0].toFixed(2)})` : 'v₁',
                    centerX + v1x * length1 * scale + 10, centerY - v1y * length1 * scale - 10);

                // Eigenvector 2 (shorter)
                const v2x = EIGENVECTORS[1][0];
                const v2y = EIGENVECTORS[1][1];
                const length2 = showTransformed ? EIGENVALUES[1] : 1;

                ctx.strokeStyle = '#7030a0';
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(centerX + v2x * length2 * scale, centerY - v2y * length2 * scale);
                ctx.stroke();

                drawArrowHead(ctx, centerX, centerY, centerX + v2x * length2 * scale, centerY - v2y * length2 * scale, '#7030a0');

                ctx.fillStyle = '#7030a0';
                ctx.fillText(showTransformed ? `λ₂v₂ (λ₂=${EIGENVALUES[1].toFixed(2)})` : 'v₂',
                    centerX + v2x * length2 * scale - 120, centerY - v2y * length2 * scale + 20);
            }
        };

        draw();
    }, [showTransformed, showEigenvectors]);

    const drawArrowHead = (ctx, fromX, fromY, toX, toY, color) => {
        const headLength = 15;
        const angle = Math.atan2(toY - fromY, toX - fromX);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    };

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-3">Geometric Intuition</h2>

            <canvas
                ref={canvasRef}
                width={500}
                height={400}
                className="border-2 border-gray-300 rounded-lg bg-white shadow-lg"
            />

            <div className="mt-4 space-y-2 w-full max-w-md">
                <button
                    onClick={() => setShowTransformed(!showTransformed)}
                    className={`w-full px-4 py-3 font-bold rounded-lg transition-colors ${showTransformed
                            ? 'bg-orange-500 hover:bg-orange-600 text-white'
                            : 'bg-blue-500 hover:bg-blue-600 text-white'
                        }`}
                >
                    {showTransformed ? '🔄 Show Original Circle' : '✨ Apply Transformation (A)'}
                </button>

                <button
                    onClick={() => setShowEigenvectors(!showEigenvectors)}
                    className={`w-full px-4 py-3 font-bold rounded-lg transition-colors ${showEigenvectors
                            ? 'bg-purple-600 hover:bg-purple-700 text-white'
                            : 'bg-green-500 hover:bg-green-600 text-white'
                        }`}
                >
                    {showEigenvectors ? '📐 Hide Eigenvectors' : '📐 Show Eigenvectors'}
                </button>
            </div>

            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200 w-full">
                <p className="text-sm text-blue-900">
                    <strong>Key Insight:</strong> When matrix A transforms the unit circle into an ellipse,
                    the ellipse axes align with the <strong>eigenvectors</strong>. The axis lengths are the
                    <strong>eigenvalues</strong>! Eigenvectors only stretch, never rotate.
                </p>
            </div>

            <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200 w-full">
                <p className="text-xs text-green-800 text-center">
                    Matrix: A = [[3, 1], [1, 2]] | Eigenvalues: λ₁ = 3.62, λ₂ = 1.38
                </p>
            </div>
        </div>
    );
}
