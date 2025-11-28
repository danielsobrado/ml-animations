import React, { useRef, useEffect } from 'react';

const ALPHA = 0.01;

export default function LeakyReluGraphPanel({ zValue = null, leakyReluValue = null, isActive = false }) {
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

    // Graph parameters
    const padding = 50;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;
    const originX = padding + graphWidth / 2;
    const originY = padding + graphHeight / 2;
    
    // Scale: 1 unit = 30 pixels
    const scale = 30;
    const xRange = 8; // -8 to 8
    const yRange = 8;

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let x = -xRange; x <= xRange; x++) {
      const px = originX + x * scale;
      ctx.beginPath();
      ctx.moveTo(px, padding);
      ctx.lineTo(px, height - padding);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let y = -yRange; y <= yRange; y++) {
      const py = originY - y * scale;
      ctx.beginPath();
      ctx.moveTo(padding, py);
      ctx.lineTo(width - padding, py);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, originY);
    ctx.lineTo(width - padding, originY);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(originX, padding);
    ctx.lineTo(originX, height - padding);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('z', width - padding + 20, originY + 5);
    ctx.fillText('f(z)', originX, padding - 10);

    // Axis tick marks and numbers
    ctx.font = '12px Arial';
    for (let x = -xRange; x <= xRange; x += 2) {
      if (x !== 0) {
        const px = originX + x * scale;
        ctx.fillText(x.toString(), px, originY + 20);
      }
    }
    for (let y = -yRange; y <= yRange; y += 2) {
      if (y !== 0) {
        const py = originY - y * scale;
        ctx.fillText(y.toString(), originX - 20, py + 4);
      }
    }
    ctx.fillText('0', originX - 15, originY + 15);

    // Draw Leaky ReLU function
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    // Draw from left side (negative x: f(x) = αx)
    for (let x = -xRange; x <= 0; x += 0.1) {
      const y = ALPHA * x;
      const px = originX + x * scale;
      const py = originY - y * scale;
      
      if (x === -xRange) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    
    // Continue to positive side (f(x) = x)
    for (let x = 0; x <= xRange; x += 0.1) {
      const y = x;
      const px = originX + x * scale;
      const py = originY - y * scale;
      ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Draw y = x reference line (dashed)
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let x = -xRange; x <= xRange; x += 0.1) {
      const y = x;
      const px = originX + x * scale;
      const py = originY - y * scale;
      
      if (x === -xRange) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Add formula label
    ctx.fillStyle = '#ea580c';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('f(z) = z if z > 0', width - 180, 30);
    ctx.fillText(`f(z) = ${ALPHA}z if z ≤ 0`, width - 180, 50);

    // Draw comparison label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Arial';
    ctx.fillText('y = z (reference)', width - 180, 70);

    // Draw current point if active
    if (isActive && zValue !== null && leakyReluValue !== null) {
      const px = originX + zValue * scale;
      const py = originY - leakyReluValue * scale;
      
      // Clamp to graph bounds
      const clampedPx = Math.max(padding, Math.min(width - padding, px));
      const clampedPy = Math.max(padding, Math.min(height - padding, py));

      // Draw vertical line from x-axis to point
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(clampedPx, originY);
      ctx.lineTo(clampedPx, clampedPy);
      ctx.stroke();
      
      // Draw horizontal line from y-axis to point
      ctx.beginPath();
      ctx.moveTo(originX, clampedPy);
      ctx.lineTo(clampedPx, clampedPy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw point
      ctx.fillStyle = '#dc2626';
      ctx.beginPath();
      ctx.arc(clampedPx, clampedPy, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw white inner circle
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(clampedPx, clampedPy, 4, 0, Math.PI * 2);
      ctx.fill();

      // Label the point
      ctx.fillStyle = '#dc2626';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'left';
      const labelX = clampedPx + 12;
      const labelY = clampedPy - 12;
      ctx.fillText(`(${zValue}, ${leakyReluValue.toFixed(2)})`, labelX, labelY);
    }

    // Add "Leaky ReLU" title in the graph
    ctx.fillStyle = '#f97316';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Leaky ReLU', originX, padding - 25);

  }, [zValue, leakyReluValue, isActive]);

  return (
    <div className="flex flex-col items-center">
      <canvas 
        ref={canvasRef} 
        width={600} 
        height={400}
        className="border border-gray-200 rounded-lg bg-white"
      />
      <div className="mt-3 text-center">
        {isActive && zValue !== null ? (
          <p className="text-gray-700">
            Current point: <span className="font-mono font-bold text-red-600">z = {zValue}</span>
            {' → '}
            <span className="font-mono font-bold text-orange-600">
              Leaky ReLU(z) = {leakyReluValue?.toFixed(2)}
            </span>
          </p>
        ) : (
          <p className="text-gray-500">
            Complete steps in the animation or practice to see the point on the graph
          </p>
        )}
      </div>
      <div className="mt-2 p-3 bg-orange-50 rounded-lg border border-orange-200 max-w-md">
        <p className="text-sm text-gray-700 text-center">
          <strong>Leaky ReLU</strong> allows a small gradient (α = {ALPHA}) when z ≤ 0,
          helping to avoid the "dying ReLU" problem where neurons stop learning.
        </p>
      </div>
    </div>
  );
}
