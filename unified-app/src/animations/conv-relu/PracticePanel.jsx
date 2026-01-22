import React, { useState } from 'react';

// Generate random small matrices for practice
function generateProblem() {
  // Input matrix X (2x3)
  const X = Array.from({ length: 2 }, () =>
    Array.from({ length: 3 }, () => Math.floor(Math.random() * 5) - 1)
  );

  // Weight matrix W (3x3)
  const W = Array.from({ length: 3 }, () =>
    Array.from({ length: 3 }, () => Math.floor(Math.random() * 5) - 2)
  );

  // Compute Z = X × W (2x3)
  const Z = X.map(row =>
    W[0].map((_, j) =>
      row.reduce((sum, x, k) => sum + x * W[k][j], 0)
    )
  );

  // Apply ReLU
  const A = Z.map(row => row.map(v => Math.max(0, v)));

  return { X, W, Z, A };
}

export default function PracticePanel() {
  const [problem, setProblem] = useState(generateProblem);
  const [currentStep, setCurrentStep] = useState(1);
  const [userZ, setUserZ] = useState(
    Array.from({ length: 2 }, () => Array(3).fill(''))
  );
  const [userA, setUserA] = useState(
    Array.from({ length: 2 }, () => Array(3).fill(''))
  );
  const [feedback, setFeedback] = useState({ z: null, a: null });
  const [showHint, setShowHint] = useState(false);

  const { X, W, Z, A } = problem;

  const handleZChange = (row, col, value) => {
    const newZ = userZ.map((r, i) =>
      r.map((v, j) => (i === row && j === col ? value : v))
    );
    setUserZ(newZ);
  };

  const handleAChange = (row, col, value) => {
    const newA = userA.map((r, i) =>
      r.map((v, j) => (i === row && j === col ? value : v))
    );
    setUserA(newA);
  };

  const checkZ = () => {
    const correct = userZ.every((row, i) =>
      row.every((val, j) => parseInt(val) === Z[i][j])
    );
    setFeedback(prev => ({ ...prev, z: correct }));
    if (correct) setCurrentStep(2);
  };

  const checkA = () => {
    const correct = userA.every((row, i) =>
      row.every((val, j) => parseInt(val) === A[i][j])
    );
    setFeedback(prev => ({ ...prev, a: correct }));
    if (correct) setCurrentStep(3);
  };

  const newProblem = () => {
    setProblem(generateProblem());
    setCurrentStep(1);
    setUserZ(Array.from({ length: 2 }, () => Array(3).fill('')));
    setUserA(Array.from({ length: 2 }, () => Array(3).fill('')));
    setFeedback({ z: null, a: null });
    setShowHint(false);
  };

  const renderInputMatrix = (matrix, label, color) => {
    const colorClasses = {
      blue: 'bg-blue-100 border-blue-400',
      green: 'bg-green-100 border-green-400',
    };
    return (
      <div className="inline-block">
        <p className={`text-sm font-bold text-center mb-1 text-${color}-700`}>{label}</p>
        <div
          className="inline-grid gap-0.5"
          style={{ gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)` }}
        >
          {matrix.flat().map((val, idx) => (
            <div
              key={idx}
              className={`w-8 h-8 flex items-center justify-center text-sm font-mono font-bold border-2 rounded ${colorClasses[color]} ${val < 0 ? 'text-red-700' : 'text-gray-800'}`}
            >
              {val}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderEditableMatrix = (values, onChange, label, color, disabled = false) => {
    const colorClasses = {
      purple: 'border-purple-400 focus:ring-purple-500',
      orange: 'border-orange-400 focus:ring-orange-500',
    };
    return (
      <div className="inline-block">
        <p className={`text-sm font-bold text-center mb-1 text-${color}-700`}>{label}</p>
        <div
          className="inline-grid gap-0.5"
          style={{ gridTemplateColumns: `repeat(${values[0].length}, 1fr)` }}
        >
          {values.flat().map((val, idx) => {
            const row = Math.floor(idx / values[0].length);
            const col = idx % values[0].length;
            return (
              <input
                key={idx}
                type="number"
                value={val}
                onChange={(e) => onChange(row, col, e.target.value)}
                disabled={disabled}
                className={`w-10 h-10 text-center text-sm font-mono font-bold border-2 rounded ${colorClasses[color]} disabled:bg-gray-100 focus:outline-none focus:ring-2`}
              />
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Interactive Practice</h2>

      {/* Problem Display */}
      <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <p className="text-center text-gray-700 mb-3 font-semibold">
          Compute: A = ReLU(X × W)
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {renderInputMatrix(X, 'X (2×3)', 'blue')}
          <span className="text-xl font-bold text-gray-800 dark:text-gray-400">×</span>
          {renderInputMatrix(W, 'W (3×3)', 'green')}
        </div>
      </div>

      {/* Step 1: Compute Z = X × W */}
      <div className={`p-4 rounded-lg border mb-3 ${currentStep >= 1 ? 'bg-white border-purple-300' : 'bg-gray-100 border-gray-200'}`}>
        <div className="flex justify-between items-center mb-2">
          <p className="font-semibold text-purple-700">Step 1: Compute Z = X × W</p>
          <button
            onClick={() => setShowHint(!showHint)}
            className="text-sm text-purple-600 hover:underline"
          >
            {showHint ? 'Hide Hint' : 'Show Hint'}
          </button>
        </div>

        {showHint && (
          <div className="mb-3 p-2 bg-purple-50 rounded text-sm text-gray-800 dark:text-gray-600">
            <p>Z[i][j] = sum of X[i][k] × W[k][j] for all k</p>
            <p className="mt-1">
              Example: Z[0][0] = {X[0][0]}×{W[0][0]} + {X[0][1]}×{W[1][0]} + {X[0][2]}×{W[2][0]} = {Z[0][0]}
            </p>
          </div>
        )}

        <div className="flex justify-center">
          {renderEditableMatrix(userZ, handleZChange, 'Z (2×3)', 'purple', feedback.z === true)}
        </div>

        <div className="mt-3 flex justify-center">
          <button
            onClick={checkZ}
            disabled={feedback.z === true}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            Check Z
          </button>
        </div>

        {feedback.z !== null && (
          <p className={`mt-2 text-center text-sm ${feedback.z ? 'text-green-600' : 'text-red-600'}`}>
            {feedback.z ? '✓ Correct!' : '✗ Try again. Check your matrix multiplication.'}
          </p>
        )}
      </div>

      {/* Step 2: Apply ReLU */}
      <div className={`p-4 rounded-lg border mb-3 ${currentStep >= 2 ? 'bg-white border-orange-300' : 'bg-gray-100 border-gray-200'}`}>
        <p className="font-semibold text-orange-700 mb-2">Step 2: Apply ReLU(Z)</p>
        <p className="text-sm text-gray-800 dark:text-gray-600 mb-3">ReLU(z) = max(0, z) — Replace negatives with 0</p>

        {currentStep >= 2 && (
          <>
            <div className="flex items-center justify-center gap-3 mb-3">
              <div className="text-center">
                <p className="text-sm font-semibold text-purple-700 mb-1">Z (from Step 1)</p>
                <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                  {Z.flat().map((val, idx) => (
                    <div
                      key={idx}
                      className={`w-8 h-8 flex items-center justify-center text-sm font-mono font-bold border-2 rounded bg-purple-100 border-purple-400 ${val < 0 ? 'text-red-700' : 'text-gray-800'}`}
                    >
                      {val}
                    </div>
                  ))}
                </div>
              </div>
              <span className="text-xl font-bold text-orange-500">→ φ →</span>
              {renderEditableMatrix(userA, handleAChange, 'A (2×3)', 'orange', feedback.a === true)}
            </div>

            <div className="flex justify-center">
              <button
                onClick={checkA}
                disabled={feedback.a === true}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50"
              >
                Check A
              </button>
            </div>

            {feedback.a !== null && (
              <p className={`mt-2 text-center text-sm ${feedback.a ? 'text-green-600' : 'text-red-600'}`}>
                {feedback.a ? '✓ Correct!' : '✗ Try again. Remember: max(0, negative) = 0'}
              </p>
            )}
          </>
        )}
      </div>

      {/* Success */}
      {currentStep === 3 && (
        <div className="p-4 bg-green-100 rounded-lg border border-green-300 text-center">
          <p className="text-green-700 font-bold text-lg mb-2">🎉 Excellent Work!</p>
          <p className="text-green-600 mb-3">
            You correctly computed the matrix multiplication and ReLU activation!
          </p>
          <button
            onClick={newProblem}
            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Try Another Problem
          </button>
        </div>
      )}

      {currentStep < 3 && (
        <button
          onClick={newProblem}
          className="w-full mt-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
        >
          Skip / New Problem
        </button>
      )}
    </div>
  );
}
