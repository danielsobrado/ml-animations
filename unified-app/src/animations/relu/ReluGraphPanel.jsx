import React, { useEffect, useRef } from 'react';

export default function ReluGraphPanel({ zValue = null, reluValue = null, isActive = false }) {
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
    const xMin = -10;
    const xMax = 10;
    const yMin = 0;
    const yMax = 10;

    // Helper functions to convert coordinates
    const toCanvasX = (x) => padding + ((x - xMin) / (xMax - xMin)) * graphWidth;
    const toCanvasY = (y) => height - padding - ((y - yMin) / (yMax - yMin)) * graphHeight;

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let x = xMin; x <= xMax; x += 2.5) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x), padding);
      ctx.lineTo(toCanvasX(x), height - padding);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let y = yMin; y <= yMax; y += 2.5) {
      ctx.beginPath();
      ctx.moveTo(padding, toCanvasY(y));
      ctx.lineTo(width - padding, toCanvasY(y));
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, toCanvasY(0));
    ctx.lineTo(width - padding, toCanvasY(0));
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), padding);
    ctx.lineTo(toCanvasX(0), height - padding);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let x = xMin; x <= xMax; x += 2.5) {
      if (x !== 0) {
        ctx.fillText(x.toString(), toCanvasX(x), toCanvasY(0) + 20);
      }
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let y = yMin; y <= yMax; y += 2.5) {
      if (y !== 0) {
        ctx.fillText(y.toString(), toCanvasX(0) - 10, toCanvasY(y) + 5);
      }
    }

    // Origin label
    ctx.textAlign = 'right';
    ctx.fillText('0', toCanvasX(0) - 10, toCanvasY(0) + 20);

    // Axis titles
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('x (input z)', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('ReLU(x)', 0, 0);
    ctx.restore();

    // Draw ReLU function
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    // Flat part (x < 0)
    ctx.moveTo(toCanvasX(xMin), toCanvasY(0));
    ctx.lineTo(toCanvasX(0), toCanvasY(0));

    // Linear part (x >= 0)
    ctx.lineTo(toCanvasX(yMax), toCanvasY(yMax));
    ctx.stroke();

    // Draw legend
    ctx.fillStyle = '#3b82f6';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('— ReLU(x)', padding + 10, padding + 20);

    // Draw current point if values provided
    if (zValue !== null && reluValue !== null && isActive) {
      const pointX = Math.max(xMin, Math.min(xMax, zValue));
      const pointY = Math.max(yMin, Math.min(yMax, reluValue));

      // Draw vertical dashed line from x-axis to point
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(toCanvasX(pointX), toCanvasY(0));
      ctx.lineTo(toCanvasX(pointX), toCanvasY(pointY));
      ctx.stroke();

      // Draw horizontal dashed line from y-axis to point
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(pointY));
      ctx.lineTo(toCanvasX(pointX), toCanvasY(pointY));
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw the point
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(toCanvasX(pointX), toCanvasY(pointY), 8, 0, 2 * Math.PI);
      ctx.fill();

      // Draw point border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label the point
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'left';
      const labelX = toCanvasX(pointX) + 15;
      const labelY = toCanvasY(pointY) - 10;
      ctx.fillText(`(${zValue}, ${reluValue})`, labelX, labelY);

      // Show input/output values
      ctx.fillStyle = '#374151';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`z = ${zValue}`, toCanvasX(pointX), toCanvasY(0) + 35);
    }

    // Title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('ReLU Activation Function', width / 2, 25);

  }, [zValue, reluValue, isActive]);

  return (
    <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-lg">
      <canvas 
        ref={canvasRef} 
        width={500} 
        height={350}
        className="border border-gray-200 rounded"
      />
      <div className="mt-3 text-center">
        <p className="text-gray-700 font-medium">
          ReLU(x) = max(0, x)
        </p>
        <p className="text-gray-500 text-sm mt-1">
          {isActive && zValue !== null ? (
            <>
              Input z = <span className="font-bold text-blue-600">{zValue}</span> → 
              Output = <span className="font-bold text-red-600">{reluValue}</span>
              {zValue < 0 && <span className="text-orange-500 ml-2">(negative → 0)</span>}
              {zValue >= 0 && <span className="text-green-500 ml-2">(positive → unchanged)</span>}
            </>
          ) : (
            'Complete the steps above to see the point on the graph'
          )}
        </p>
      </div>
    </div>
  );
}
