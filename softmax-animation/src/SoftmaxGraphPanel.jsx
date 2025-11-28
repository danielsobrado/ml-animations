import React, { useEffect, useRef } from 'react';

export default function SoftmaxGraphPanel({ logits = [], probabilities = [], isActive = false }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        if (!isActive || logits.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Start animation to see graph', width / 2, height / 2);
            return;
        }

        const padding = 50;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;

        // Bar settings
        const barWidth = 40;
        const gap = (chartWidth - (logits.length * barWidth)) / (logits.length + 1);

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;

        // Y-axis (Probabilities 0-1)
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();

        // X-axis (Classes)
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Y-axis labels
        ctx.fillStyle = '#333';
        ctx.textAlign = 'right';
        ctx.font = '12px Arial';
        for (let i = 0; i <= 10; i += 2) {
            const val = i / 10;
            const y = height - padding - (val * chartHeight);
            ctx.fillText(val.toFixed(1), padding - 5, y + 4);

            // Grid line
            ctx.strokeStyle = '#eee';
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        // Draw Bars
        logits.forEach((logit, i) => {
            const prob = probabilities[i] || 0;
            const x = padding + gap + i * (barWidth + gap);
            const barHeight = prob * chartHeight;
            const y = height - padding - barHeight;

            // Bar
            ctx.fillStyle = '#ed7d31'; // Orange for prob
            ctx.fillRect(x, y, barWidth, barHeight);

            // Logit value (below axis)
            ctx.fillStyle = '#5b9bd5'; // Blue for logit
            ctx.textAlign = 'center';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(`z=${logit.toFixed(1)}`, x + barWidth / 2, height - padding + 20);

            // Prob value (above bar)
            ctx.fillStyle = '#ed7d31';
            ctx.fillText(prob.toFixed(2), x + barWidth / 2, y - 5);
        });

        // Title
        ctx.fillStyle = '#333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Probabilities Distribution', width / 2, 30);

    }, [logits, probabilities, isActive]);

    return (
        <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-lg">
            <canvas
                ref={canvasRef}
                width={400}
                height={300}
                className="border border-gray-200 rounded"
            />
            <div className="mt-3 text-center">
                <p className="text-gray-700 font-medium">
                    Softmax(z) = e^z / Σe^z
                </p>
            </div>
        </div>
    );
}
