import React, { useRef, useEffect, useState } from 'react';

const MATRIX_A = [[3, 1], [1, 2]];
const EIGENVECTORS = [[0.85, 0.53], [-0.53, 0.85]];
const EIGENVALUES = [3.62, 1.38];

export default function InteractiveExplorerPanel() {
    const canvasRef = useRef(null);
    const [vectorX, setVectorX] = useState(1);
    const [vectorY, setVectorY] = useState(0);
    const [isDragging, setIsDragging] = useState(false);

    const scale = 60;
    const centerX = 250;
    const centerY = 200;

    // Transform vector by matrix A
    const transformedX = MATRIX_A[0][0] * vectorX + MATRIX_A[0][1] * vectorY;
    const transformedY = MATRIX_A[1][0] * vectorX + MATRIX_A[1][1] * vectorY;

    // Calculate magnitudes
    const originalMag = Math.sqrt(vectorX ** 2 + vectorY ** 2);
    const transformedMag = Math.sqrt(transformedX ** 2 + transformedY ** 2);
    const scaleFactor = originalMag > 0.01 ? transformedMag / originalMag : 0;

    // Check if close to eigenvector direction
    const isNearEigenvector = (threshold = 0.15) => {
        const vNorm = Math.sqrt(vectorX ** 2 + vectorY ** 2);
        if (vNorm < 0.1) return false;

        const vNormalized = [vectorX / vNorm, vectorY / vNorm];

        for (let i = 0; i < EIGENVECTORS.length; i++) {
            const ev = EIGENVECTORS[i];
            const dotProduct = Math.abs(vNormalized[0] * ev[0] + vNormalized[1] * ev[1]);
            if (dotProduct > 1 - threshold) {
                return i + 1; // Return which eigenvector (1 or 2)
            }
        }
        return false;
    };

    const nearEigen = isNearEigenvector();

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const draw = () => {
            // Clear
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Grid
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            for (let i = -4; i <= 4; i++) {
                ctx.beginPath();
                ctx.moveTo(centerX + i * scale, 0);
                ctx.lineTo(centerX + i * scale, canvas.height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, centerY + i * scale);
                ctx.lineTo(canvas.width, centerY + i * scale);
                ctx.stroke();
            }

            // Axes
            ctx.strokeStyle = '#cccccc';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(canvas.width, centerY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(centerX, 0);
            ctx.lineTo(centerX, canvas.height);
            ctx.stroke();

            // Draw eigenvector lines (faint)
            ctx.strokeStyle = '#d0d0d0';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);

            // Eigenvector 1 line
            const ev1 = EIGENVECTORS[0];
            ctx.beginPath();
            ctx.moveTo(centerX - ev1[0] * 300, centerY + ev1[1] * 300);
            ctx.lineTo(centerX + ev1[0] * 300, centerY - ev1[1] * 300);
            ctx.stroke();

            // Eigenvector 2 line
            const ev2 = EIGENVECTORS[1];
            ctx.beginPath();
            ctx.moveTo(centerX - ev2[0] * 300, centerY + ev2[1] * 300);
            ctx.lineTo(centerX + ev2[0] * 300, centerY - ev2[1] * 300);
            ctx.stroke();

            ctx.setLineDash([]);

            // Draw transformed vector (Av)
            drawVector(ctx, 0, 0, transformedX, transformedY, '#ed7d31', 'Av', 3);

            // Draw original vector (v)
            drawVector(ctx, 0, 0, vectorX, vectorY, '#5b9bd5', 'v', 4);

            // Highlight if near eigenvector
            if (nearEigen) {
                ctx.fillStyle = nearEigen === 1 ? '#70ad47' : '#7030a0';
                ctx.font = 'bold 18px Arial';
                ctx.fillText(`🎯 On Eigenvector ${nearEigen}!`, 10, 30);
                ctx.font = '14px Arial';
                ctx.fillText(`Both vectors align!`, 10, 50);
            }
        };

        draw();
    }, [vectorX, vectorY, transformedX, transformedY, nearEigen]);

    const drawVector = (ctx, fromX, fromY, toX, toY, color, label, lineWidth) => {
        const screenFromX = centerX + fromX * scale;
        const screenFromY = centerY - fromY * scale;
        const screenToX = centerX + toX * scale;
        const screenToY = centerY - toY * scale;

        // Vector line
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(screenFromX, screenFromY);
        ctx.lineTo(screenToX, screenToY);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(screenToY - screenFromY, screenToX - screenFromX);
        const headLength = 15;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(screenToX, screenToY);
        ctx.lineTo(
            screenToX - headLength * Math.cos(angle - Math.PI / 6),
            screenToY - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            screenToX - headLength * Math.cos(angle + Math.PI / 6),
            screenToY - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();

        // Label
        ctx.fillStyle = color;
        ctx.font = 'bold 16px Arial';
        ctx.fillText(label, screenToX + 10, screenToY - 10);
    };

    const handleMouseDown = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const vScreenX = centerX + vectorX * scale;
        const vScreenY = centerY - vectorY * scale;

        const dist = Math.sqrt((mouseX - vScreenX) ** 2 + (mouseY - vScreenY) ** 2);
        if (dist < 20) {
            setIsDragging(true);
        }
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const newX = (mouseX - centerX) / scale;
        const newY = -(mouseY - centerY) / scale;

        // Clamp to reasonable range
        const maxLen = 3;
        const len = Math.sqrt(newX ** 2 + newY ** 2);
        if (len > maxLen) {
            setVectorX((newX / len) * maxLen);
            setVectorY((newY / len) * maxLen);
        } else {
            setVectorX(newX);
            setVectorY(newY);
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-3">Interactive Vector Explorer</h2>

            <canvas
                ref={canvasRef}
                width={500}
                height={400}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                className="border-2 border-gray-300 rounded-lg bg-white shadow-lg cursor-pointer"
            />

            <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-orange-50 rounded-lg border-2 border-gray-300 w-full">
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-blue-100 p-3 rounded-lg border border-blue-300">
                        <p className="text-sm font-bold text-blue-900">Original Vector (v)</p>
                        <p className="text-xs text-blue-800 mt-1">
                            v = [{vectorX.toFixed(2)}, {vectorY.toFixed(2)}]
                        </p>
                        <p className="text-xs text-blue-800">||v|| = {originalMag.toFixed(2)}</p>
                    </div>

                    <div className="bg-orange-100 p-3 rounded-lg border border-orange-300">
                        <p className="text-sm font-bold text-orange-900">Transformed (Av)</p>
                        <p className="text-xs text-orange-800 mt-1">
                            Av = [{transformedX.toFixed(2)}, {transformedY.toFixed(2)}]
                        </p>
                        <p className="text-xs text-orange-800">||Av|| = {transformedMag.toFixed(2)}</p>
                    </div>
                </div>

                <div className="mt-3 p-3 bg-yellow-100 rounded-lg border border-yellow-400 text-center">
                    <p className="text-sm font-bold text-yellow-900">
                        Scale Factor: ||Av|| / ||v|| = {scaleFactor.toFixed(2)}
                        {nearEigen && ` ≈ λ${nearEigen} = ${EIGENVALUES[nearEigen - 1].toFixed(2)}`}
                    </p>
                </div>
            </div>

            <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200 w-full">
                <p className="text-xs text-green-800 text-center">
                    <strong>💡 Try this:</strong> Drag the blue vector (v). When it aligns with the faint lines
                    (eigenvector directions), both v and Av point the same way—only the length changes!
                </p>
            </div>

            <div className="mt-3 space-x-2">
                <button
                    onClick={() => { setVectorX(EIGENVECTORS[0][0]); setVectorY(EIGENVECTORS[0][1]); }}
                    className="px-3 py-2 bg-green-500 hover:bg-green-600 text-white font-bold rounded-lg text-sm"
                >
                    Set to v₁
                </button>
                <button
                    onClick={() => { setVectorX(EIGENVECTORS[1][0]); setVectorY(EIGENVECTORS[1][1]); }}
                    className="px-3 py-2 bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg text-sm"
                >
                    Set to v₂
                </button>
                <button
                    onClick={() => { setVectorX(1); setVectorY(0); }}
                    className="px-3 py-2 bg-gray-500 hover:bg-gray-600 text-white font-bold rounded-lg text-sm"
                >
                    Reset
                </button>
            </div>
        </div>
    );
}
