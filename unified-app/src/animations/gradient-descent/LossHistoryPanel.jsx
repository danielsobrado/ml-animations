import React, { useEffect, useRef } from 'react';

export default function LossHistoryPanel({ history = [] }) {
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

        if (history.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Run to see loss history', width / 2, height / 2);
            return;
        }

        const padding = 50;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;

        // Find max loss for scaling
        const maxLoss = Math.max(...history.map(h => h.loss), 1);
        const maxIter = history.length - 1;

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';

        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Iteration', width / 2, height - 10);

        // Y-axis label
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Loss', 0, 0);
        ctx.restore();

        // Plot line
        ctx.strokeStyle = '#7030a0';
        ctx.lineWidth = 2;
        ctx.beginPath();

        history.forEach((point, i) => {
            const x = padding + (i / maxIter) * chartWidth;
            const y = height - padding - (point.loss / maxLoss) * chartHeight;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Plot points
        ctx.fillStyle = '#5b9bd5';
        history.forEach((point, i) => {
            const x = padding + (i / maxIter) * chartWidth;
            const y = height - padding - (point.loss / maxLoss) * chartHeight;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });

    }, [history]);

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
                    Loss over Iterations
                </p>
            </div>
        </div>
    );
}
