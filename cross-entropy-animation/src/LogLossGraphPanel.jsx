import React, { useEffect, useRef } from 'react';

export default function LogLossGraphPanel({ probability = 0.7, isActive = false }) {
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

        const padding = 50;
        const graphWidth = width - 2 * padding;
        const graphHeight = height - 2 * padding;

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;

        // Y-axis (Loss)
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();

        // X-axis (Probability)
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';

        // X labels
        for (let i = 0; i <= 10; i += 2) {
            const val = i / 10;
            const x = padding + (val * graphWidth);
            ctx.fillText(val.toFixed(1), x, height - padding + 20);
        }
        ctx.fillText('Probability (p)', width / 2, height - 10);

        // Y labels (Loss goes from 0 to ~5 for visualization)
        ctx.textAlign = 'right';
        const maxLoss = 5;
        for (let i = 0; i <= maxLoss; i++) {
            const y = height - padding - (i / maxLoss * graphHeight);
            ctx.fillText(i.toString(), padding - 10, y + 4);
        }
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Loss (-ln(p))', 0, 0);
        ctx.restore();

        // Plot Curve y = -ln(x)
        ctx.strokeStyle = '#ed7d31';
        ctx.lineWidth = 3;
        ctx.beginPath();

        // Start from small epsilon to avoid infinity
        const epsilon = 0.01;
        for (let x = epsilon; x <= 1.0; x += 0.01) {
            const loss = -Math.log(x);
            if (loss > maxLoss) continue;

            const canvasX = padding + (x * graphWidth);
            const canvasY = height - padding - (loss / maxLoss * graphHeight);

            if (x === epsilon) ctx.moveTo(canvasX, canvasY);
            else ctx.lineTo(canvasX, canvasY);
        }
        ctx.stroke();

        // Draw Current Point
        if (isActive) {
            const loss = -Math.log(probability);
            const clampedLoss = Math.min(loss, maxLoss);

            const pointX = padding + (probability * graphWidth);
            const pointY = height - padding - (clampedLoss / maxLoss * graphHeight);

            // Dashed lines
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);

            ctx.beginPath();
            ctx.moveTo(pointX, height - padding);
            ctx.lineTo(pointX, pointY);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(padding, pointY);
            ctx.lineTo(pointX, pointY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Point
            ctx.fillStyle = '#5b9bd5';
            ctx.beginPath();
            ctx.arc(pointX, pointY, 6, 0, 2 * Math.PI);
            ctx.fill();

            // Text
            ctx.fillStyle = '#333';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(`p=${probability.toFixed(2)}`, pointX + 10, pointY - 10);
            ctx.fillText(`L=${loss.toFixed(2)}`, pointX + 10, pointY + 10);
        }

    }, [probability, isActive]);

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
                    Log Loss Curve: y = -ln(x)
                </p>
            </div>
        </div>
    );
}
