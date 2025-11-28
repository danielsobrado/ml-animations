import React, { useState } from 'react';

// Data from the diagram
// Input X (3x3):
// [[2, 1, 0],
//  [1, 1, 1],
//  [3, 0, 1]]

// Three weight matrices W1, W2, W3 (4x4 each):
// W1: [[1, -1, 1, -5], [1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, -2]]
// W2: [[-1, -5, -5, ?], [3, 2, 1, ?], [5, 2, 3, ?], [3, -1, -1, ?]]
// W3: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] (biases shown as 1,1,1 at bottom)

// After Layer 1 (X × W + b → ReLU):
// Z1 matrix (4x3):
// [[-1, -5, -5], [3, 2, 1], [5, 2, 3], [3, -1, -1]]
// After ReLU: [[0, 0, 0], [3, 2, 1], [5, 2, 3], [3, 0, 0]]

// Layer 2 weights (2x4):
// W_L2: [[1, 1, -1, 0, 0], [0, 0, 1, -1, 1]]
// Z2: [[-2, 0, -2], [3, 3, 4]]
// After ReLU: [[0, 0, 0], [3, 3, 4]]

export default function AnimationPanel() {
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Input matrix X (3x3)
  const X = [
    [2, 1, 0],
    [1, 1, 1],
    [3, 0, 1]
  ];

  // Weight matrix W (4x4) - combining the weights shown
  const W = [
    [1, -1, 1, -5],
    [1, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 1, -2]
  ];

  // Bias for layer 1 (3 outputs, using column biases from diagram)
  const b1 = [-1, -5, -5];

  // Pre-ReLU output Z1 (4x3) - as shown in diagram
  const Z1 = [
    [-1, -5, -5],
    [3, 2, 1],
    [5, 2, 3],
    [3, -1, -1]
  ];

  // After ReLU (4x3)
  const A1 = [
    [0, 0, 0],
    [3, 2, 1],
    [5, 2, 3],
    [3, 0, 0]
  ];

  // Layer 2 weights (2x4)
  const W2 = [
    [1, 1, -1, 0],
    [0, 0, 1, -1]
  ];

  // Bias for layer 2
  const b2 = [0, 1];

  // Pre-ReLU output Z2 (2x3)
  const Z2 = [
    [-2, 0, -2],
    [3, 3, 4]
  ];

  // After ReLU (2x3)
  const A2 = [
    [0, 0, 0],
    [3, 3, 4]
  ];

  const steps = [
    { title: 'Ready', desc: 'Click Play to start the animation' },
    { title: 'Input X', desc: 'The 3×3 input matrix' },
    { title: 'Layer 1 Weights', desc: 'Weight matrix W₁ (4×4)' },
    { title: 'Matrix Multiply', desc: 'Z₁ = X × W₁ᵀ (compute pre-activation)' },
    { title: 'Apply ReLU', desc: 'A₁ = ReLU(Z₁) = max(0, Z₁)' },
    { title: 'Layer 2 Weights', desc: 'Weight matrix W₂ (2×4)' },
    { title: 'Matrix Multiply', desc: 'Z₂ = A₁ × W₂ᵀ (compute pre-activation)' },
    { title: 'Apply ReLU', desc: 'A₂ = ReLU(Z₂) = Final Output' },
  ];

  const playAnimation = () => {
    if (isPlaying) return;
    setIsPlaying(true);
    setStep(0);

    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep++;
      if (currentStep >= steps.length) {
        clearInterval(interval);
        setIsPlaying(false);
      } else {
        setStep(currentStep);
      }
    }, 1500);
  };

  const resetAnimation = () => {
    setStep(0);
    setIsPlaying(false);
  };

  const nextStep = () => {
    if (step < steps.length - 1) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > 0) setStep(step - 1);
  };

  const renderMatrix = (matrix, label, color, highlight = false, highlightCells = []) => {
    const colorClasses = {
      blue: 'bg-blue-100 border-blue-400',
      green: 'bg-green-100 border-green-400',
      purple: 'bg-purple-100 border-purple-400',
      orange: 'bg-orange-100 border-orange-400',
      red: 'bg-red-100 border-red-400',
      yellow: 'bg-yellow-100 border-yellow-400',
    };

    return (
      <div className={`inline-block ${highlight ? 'ring-4 ring-yellow-400 rounded-lg' : ''}`}>
        <p className={`text-sm font-bold text-center mb-1 text-${color}-700`}>{label}</p>
        <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)` }}>
          {matrix.flat().map((val, idx) => {
            const row = Math.floor(idx / matrix[0].length);
            const col = idx % matrix[0].length;
            const isHighlighted = highlightCells.some(([r, c]) => r === row && c === col);
            return (
              <div
                key={idx}
                className={`w-9 h-9 flex items-center justify-center text-sm font-mono font-bold border-2 rounded
                  ${colorClasses[color]}
                  ${isHighlighted ? 'ring-2 ring-red-500 bg-red-200' : ''}
                  ${val < 0 ? 'text-red-700' : 'text-gray-800'}`}
              >
                {val}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Animation Demo</h2>

      {/* Visualization Area */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 min-h-[450px]">
        {step === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-xl text-gray-500">Click Play to start</p>
          </div>
        )}

        {step >= 1 && (
          <div className="space-y-4">
            {/* Layer 1 */}
            <div className="flex flex-wrap items-center justify-center gap-3">
              {/* Input X */}
              <div className={step === 1 ? 'animate-pulse' : ''}>
                {renderMatrix(X, 'X (Input)', 'blue', step === 1)}
              </div>

              {step >= 2 && (
                <>
                  <span className="text-2xl font-bold text-gray-400">×</span>
                  <div className={step === 2 ? 'animate-pulse' : ''}>
                    {renderMatrix(W, 'W₁ (Weights)', 'green', step === 2)}
                  </div>
                </>
              )}

              {step >= 3 && (
                <>
                  <span className="text-2xl font-bold text-gray-400">=</span>
                  <div className={step === 3 ? 'animate-pulse' : ''}>
                    {renderMatrix(Z1, 'Z₁ (Pre-ReLU)', 'purple', step === 3)}
                  </div>
                </>
              )}

              {step >= 4 && (
                <>
                  <span className="text-2xl font-bold text-orange-500">→ φ ≈</span>
                  <div className={step === 4 ? 'animate-pulse' : ''}>
                    {renderMatrix(A1, 'A₁ (After ReLU)', 'orange', step === 4)}
                  </div>
                </>
              )}
            </div>

            {/* Layer 2 */}
            {step >= 5 && (
              <div className="flex flex-wrap items-center justify-center gap-3 pt-4 border-t border-gray-200">
                <div className={step === 4 ? '' : ''}>
                  {renderMatrix(A1, 'A₁', 'orange')}
                </div>

                {step >= 5 && (
                  <>
                    <span className="text-2xl font-bold text-gray-400">×</span>
                    <div className={step === 5 ? 'animate-pulse' : ''}>
                      {renderMatrix(W2, 'W₂ (Weights)', 'green', step === 5)}
                    </div>
                  </>
                )}

                {step >= 6 && (
                  <>
                    <span className="text-2xl font-bold text-gray-400">=</span>
                    <div className={step === 6 ? 'animate-pulse' : ''}>
                      {renderMatrix(Z2, 'Z₂ (Pre-ReLU)', 'red', step === 6)}
                    </div>
                  </>
                )}

                {step >= 7 && (
                  <>
                    <span className="text-2xl font-bold text-orange-500">→ φ ≈</span>
                    <div className={step === 7 ? 'animate-pulse' : ''}>
                      {renderMatrix(A2, 'A₂ (Output)', 'yellow', step === 7)}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mt-4 flex justify-center gap-2">
        <button
          onClick={prevStep}
          disabled={step === 0 || isPlaying}
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
          disabled={step >= steps.length - 1 || isPlaying}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50"
        >
          Next →
        </button>
      </div>

      {/* Step Info */}
      <div className="mt-4 text-center">
        <p className="text-lg font-semibold text-gray-700">
          Step {step}/{steps.length - 1}: {steps[step].title}
        </p>
        <p className="text-sm text-gray-500">{steps[step].desc}</p>
        <div className="mt-2 flex justify-center gap-1">
          {steps.map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full ${i <= step ? 'bg-blue-600' : 'bg-gray-300'}`}
            />
          ))}
        </div>
      </div>

      {/* Formula */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-sm text-gray-700">
          <strong>Two-Layer Network:</strong>
        </p>
        <p className="text-sm font-mono text-gray-600 mt-1">
          Layer 1: A₁ = ReLU(X × W₁ᵀ + b₁)
        </p>
        <p className="text-sm font-mono text-gray-600">
          Layer 2: A₂ = ReLU(A₁ × W₂ᵀ + b₂)
        </p>
        <p className="text-sm text-gray-500 mt-2">
          φ (ReLU): max(0, z) — negative values become 0
        </p>
      </div>
    </div>
  );
}
