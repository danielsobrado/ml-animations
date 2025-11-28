import React, { useState, useEffect, useRef } from 'react';

export default function AnimationPanel() {
  const [step, setStep] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(800);
  const intervalRef = useRef(null);

  // 5x5 Input image/matrix
  const input = [
    [1, 2, 0, 1, 2],
    [0, 1, 3, 2, 1],
    [2, 3, 1, 0, 2],
    [1, 0, 2, 3, 1],
    [2, 1, 0, 1, 3]
  ];

  // 3x3 Kernel (edge detection style)
  const kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
  ];

  // Calculate output size: (input_size - kernel_size) / stride + 1 = (5-3)/1 + 1 = 3
  const outputSize = 3;
  
  // Compute the full output matrix
  const computeOutput = () => {
    const output = [];
    for (let i = 0; i < outputSize; i++) {
      const row = [];
      for (let j = 0; j < outputSize; j++) {
        let sum = 0;
        for (let ki = 0; ki < 3; ki++) {
          for (let kj = 0; kj < 3; kj++) {
            sum += input[i + ki][j + kj] * kernel[ki][kj];
          }
        }
        row.push(sum);
      }
      output.push(row);
    }
    return output;
  };

  const output = computeOutput();
  const totalSteps = outputSize * outputSize; // 9 positions

  // Get current kernel position based on step
  const getKernelPosition = (s) => {
    if (s < 0) return { row: -1, col: -1 };
    const row = Math.floor(s / outputSize);
    const col = s % outputSize;
    return { row, col };
  };

  const { row: kernelRow, col: kernelCol } = getKernelPosition(step);

  // Calculate current convolution value
  const getCurrentValue = () => {
    if (step < 0) return null;
    return output[kernelRow][kernelCol];
  };

  // Get element-wise products for current position
  const getProducts = () => {
    if (step < 0) return [];
    const products = [];
    for (let ki = 0; ki < 3; ki++) {
      for (let kj = 0; kj < 3; kj++) {
        const inputVal = input[kernelRow + ki][kernelCol + kj];
        const kernelVal = kernel[ki][kj];
        products.push({
          input: inputVal,
          kernel: kernelVal,
          product: inputVal * kernelVal
        });
      }
    }
    return products;
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const playAnimation = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setStep(-1);

    let currentStep = -1;
    intervalRef.current = setInterval(() => {
      currentStep++;
      if (currentStep >= totalSteps) {
        clearInterval(intervalRef.current);
        setIsPlaying(false);
      } else {
        setStep(currentStep);
      }
    }, speed);
  };

  const resetAnimation = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStep(-1);
    setIsPlaying(false);
  };

  const nextStep = () => {
    if (step < totalSteps - 1) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > -1) setStep(step - 1);
  };

  // Check if a cell is under the kernel
  const isUnderKernel = (row, col) => {
    if (step < 0) return false;
    return row >= kernelRow && row < kernelRow + 3 && col >= kernelCol && col < kernelCol + 3;
  };

  // Get kernel-relative position for coloring
  const getKernelRelativePos = (row, col) => {
    if (!isUnderKernel(row, col)) return null;
    return { ki: row - kernelRow, kj: col - kernelCol };
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Animation Demo</h2>

      {/* Main visualization */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex flex-wrap items-start justify-center gap-6">
          {/* Input Matrix */}
          <div>
            <p className="text-sm font-bold text-blue-700 text-center mb-2">Input (5×5)</p>
            <div className="relative">
              <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
                {input.flat().map((val, idx) => {
                  const row = Math.floor(idx / 5);
                  const col = idx % 5;
                  const underKernel = isUnderKernel(row, col);
                  const relPos = getKernelRelativePos(row, col);
                  
                  let bgColor = 'bg-blue-50';
                  if (underKernel) {
                    // Color based on kernel value at this position
                    const kernelVal = relPos ? kernel[relPos.ki][relPos.kj] : 0;
                    if (kernelVal > 0) bgColor = 'bg-green-200';
                    else if (kernelVal < 0) bgColor = 'bg-red-200';
                    else bgColor = 'bg-yellow-100';
                  }

                  return (
                    <div
                      key={idx}
                      className={`w-10 h-10 flex items-center justify-center text-sm font-mono font-bold border-2 rounded
                        ${bgColor}
                        ${underKernel ? 'border-purple-500 ring-1 ring-purple-300' : 'border-blue-300'}
                      `}
                    >
                      {val}
                    </div>
                  );
                })}
              </div>
              {/* Kernel overlay indicator */}
              {step >= 0 && (
                <div
                  className="absolute border-4 border-purple-600 rounded-lg pointer-events-none transition-all duration-300"
                  style={{
                    top: kernelRow * 42 - 2,
                    left: kernelCol * 42 - 2,
                    width: 3 * 42 + 4,
                    height: 3 * 42 + 4,
                  }}
                />
              )}
            </div>
          </div>

          {/* Convolution symbol */}
          <div className="flex flex-col items-center justify-center">
            <span className="text-3xl font-bold text-gray-400">∗</span>
            <span className="text-xs text-gray-500">convolve</span>
          </div>

          {/* Kernel */}
          <div>
            <p className="text-sm font-bold text-green-700 text-center mb-2">Kernel (3×3)</p>
            <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
              {kernel.flat().map((val, idx) => (
                <div
                  key={idx}
                  className={`w-10 h-10 flex items-center justify-center text-sm font-mono font-bold border-2 rounded
                    ${val > 0 ? 'bg-green-100 border-green-400' : val < 0 ? 'bg-red-100 border-red-400' : 'bg-yellow-50 border-yellow-400'}
                    ${val < 0 ? 'text-red-700' : 'text-gray-800'}
                  `}
                >
                  {val}
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 text-center mt-1">Edge detection</p>
          </div>

          {/* Equals */}
          <div className="flex items-center">
            <span className="text-3xl font-bold text-gray-400">=</span>
          </div>

          {/* Output Matrix */}
          <div>
            <p className="text-sm font-bold text-purple-700 text-center mb-2">Output (3×3)</p>
            <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
              {output.flat().map((val, idx) => {
                const row = Math.floor(idx / 3);
                const col = idx % 3;
                const isCurrent = row === kernelRow && col === kernelCol;
                const isComputed = step >= 0 && (row < kernelRow || (row === kernelRow && col <= kernelCol));

                return (
                  <div
                    key={idx}
                    className={`w-10 h-10 flex items-center justify-center text-sm font-mono font-bold border-2 rounded transition-all
                      ${isCurrent ? 'bg-purple-300 border-purple-600 ring-2 ring-purple-400 scale-110' : 
                        isComputed ? 'bg-purple-100 border-purple-400' : 'bg-gray-100 border-gray-300'}
                      ${val < 0 ? 'text-red-700' : 'text-gray-800'}
                    `}
                  >
                    {isComputed ? val : '?'}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Calculation breakdown */}
        {step >= 0 && (
          <div className="mt-4 p-3 bg-purple-50 rounded-lg border border-purple-200">
            <p className="text-sm font-semibold text-purple-700 mb-2">
              Position ({kernelRow}, {kernelCol}) — Element-wise multiply and sum:
            </p>
            <div className="flex flex-wrap gap-1 items-center text-sm font-mono">
              {getProducts().map((p, idx) => (
                <span key={idx} className="flex items-center">
                  <span className={`px-1 rounded ${p.kernel > 0 ? 'bg-green-100' : p.kernel < 0 ? 'bg-red-100' : 'bg-yellow-50'}`}>
                    {p.input}×{p.kernel}
                  </span>
                  <span className="text-gray-400 mx-0.5">=</span>
                  <span className={`px-1 rounded ${p.product < 0 ? 'text-red-600' : ''}`}>
                    {p.product}
                  </span>
                  {idx < 8 && <span className="text-gray-400 mx-1">+</span>}
                </span>
              ))}
              <span className="text-gray-600 mx-2">=</span>
              <span className="font-bold text-purple-700 text-lg">{getCurrentValue()}</span>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        <button
          onClick={prevStep}
          disabled={step <= -1 || isPlaying}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50"
        >
          ← Prev
        </button>
        <button
          onClick={playAnimation}
          disabled={isPlaying}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isPlaying ? 'Playing...' : 'Play'}
        </button>
        <button
          onClick={resetAnimation}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
        >
          Reset
        </button>
        <button
          onClick={nextStep}
          disabled={step >= totalSteps - 1 || isPlaying}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50"
        >
          Next →
        </button>
      </div>

      {/* Speed control */}
      <div className="mt-3 flex justify-center items-center gap-2">
        <span className="text-sm text-gray-600">Speed:</span>
        <input
          type="range"
          min="200"
          max="1500"
          step="100"
          value={1700 - speed}
          onChange={(e) => setSpeed(1700 - parseInt(e.target.value))}
          className="w-32"
        />
        <span className="text-sm text-gray-600">{speed}ms</span>
      </div>

      {/* Step indicator */}
      <div className="mt-3 text-center">
        <p className="text-lg font-semibold text-gray-700">
          {step < 0 ? 'Ready — Click Play' : `Step ${step + 1} / ${totalSteps}`}
        </p>
        <div className="mt-2 flex justify-center gap-1 flex-wrap">
          {Array.from({ length: totalSteps }).map((_, i) => (
            <div
              key={i}
              onClick={() => !isPlaying && setStep(i)}
              className={`w-6 h-6 rounded flex items-center justify-center text-xs cursor-pointer transition-all
                ${i === step ? 'bg-purple-600 text-white scale-110' : 
                  i < step ? 'bg-purple-300 text-purple-800' : 'bg-gray-200 text-gray-500'}
              `}
            >
              {i + 1}
            </div>
          ))}
        </div>
      </div>

      {/* Formula */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-sm text-gray-700">
          <strong>2D Convolution Formula:</strong>
        </p>
        <p className="text-sm font-mono text-gray-600 mt-1">
          Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m, n]
        </p>
        <p className="text-xs text-gray-500 mt-2">
          The kernel slides across the input, computing element-wise products and summing them at each position.
        </p>
      </div>
    </div>
  );
}
