import React, { useEffect, useRef } from 'react';

export default function GatingPanel({ numExperts, topK }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Simulate some logits
        const logits = Array.from({ length: numExperts }, () => Math.random() * 5);
        // Softmax
        const exp = logits.map(x => Math.exp(x));
        const sum = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map(x => x / sum);

        // Find top-k indices
        const indices = probs.map((p, i) => ({ p, i }))
            .sort((a, b) => b.p - a.p)
            .slice(0, topK)
            .map(x => x.i);

        // Draw
        const width = canvas.width;
        const height = canvas.height;
        const barWidth = width / numExperts;

        ctx.clearRect(0, 0, width, height);

        probs.forEach((p, i) => {
            const isSelected = indices.includes(i);
            const x = i * barWidth;
            const h = p * height * 0.8;
            const y = height - h;

            // Bar
            ctx.fillStyle = isSelected ? '#00f3ff' : '#334155';
            ctx.fillRect(x + 2, y, barWidth - 4, h);

            // Label
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px monospace';
            ctx.fillText(`E${i}`, x + barWidth / 2 - 6, height - 5);

            // Value
            if (isSelected) {
                ctx.fillStyle = '#ffffff';
                ctx.fillText(p.toFixed(2), x + 2, y - 5);
            }
        });

    }, [numExperts, topK]);

    return (
        <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
            <h3 className="text-sm font-bold text-slate-400 mb-2 uppercase">Gating Probabilities</h3>
            <canvas ref={canvasRef} width={300} height={100} className="w-full h-24" />
        </div>
    );
}
