import React, { useState } from 'react';

// Generate a random practice problem
function generateProblem() {
  // 4x4 input for simpler practice
  const input = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => Math.floor(Math.random() * 5))
  );

  // 3x3 kernel with simple values
  const kernelTypes = [
    { name: 'Edge (Vertical)', kernel: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
    { name: 'Edge (Horizontal)', kernel: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
    { name: 'Sharpen', kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    { name: 'Blur (Average)', kernel: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] },
    { name: 'Identity', kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
  ];

  const selectedKernel = kernelTypes[Math.floor(Math.random() * kernelTypes.length)];
  const kernel = selectedKernel.kernel;
  const kernelName = selectedKernel.name;

  // Output is 2x2 for 4x4 input with 3x3 kernel
  const output = [];
  for (let i = 0; i < 2; i++) {
    const row = [];
    for (let j = 0; j < 2; j++) {
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

  return { input, kernel, kernelName, output };
}

export default function PracticePanel() {
  const [problem, setProblem] = useState(generateProblem);
  const [currentCell, setCurrentCell] = useState({ row: 0, col: 0 });
  const [userAnswers, setUserAnswers] = useState([['', ''], ['', '']]);
  const [feedback, setFeedback] = useState([[null, null], [null, null]]);
  const [showHint, setShowHint] = useState(false);
  const [completed, setCompleted] = useState(false);

  const { input, kernel, kernelName, output } = problem;

  const handleAnswerChange = (row, col, value) => {
    const newAnswers = userAnswers.map((r, i) =>
      r.map((v, j) => (i === row && j === col ? value : v))
    );
    setUserAnswers(newAnswers);
  };

  const checkAnswer = (row, col) => {
    const userVal = parseInt(userAnswers[row][col]);
    const correct = userVal === output[row][col];
    
    const newFeedback = feedback.map((r, i) =>
      r.map((v, j) => (i === row && j === col ? correct : v))
    );
    setFeedback(newFeedback);

    if (correct) {
      // Move to next cell
      if (col < 1) {
        setCurrentCell({ row, col: col + 1 });
      } else if (row < 1) {
        setCurrentCell({ row: row + 1, col: 0 });
      } else {
        setCompleted(true);
      }
    }
  };

  const checkAll = () => {
    const newFeedback = output.map((row, i) =>
      row.map((val, j) => parseInt(userAnswers[i][j]) === val)
    );
    setFeedback(newFeedback);
    
    const allCorrect = newFeedback.every(row => row.every(v => v === true));
    if (allCorrect) setCompleted(true);
  };

  const newProblem = () => {
    setProblem(generateProblem());
    setCurrentCell({ row: 0, col: 0 });
    setUserAnswers([['', ''], ['', '']]);
    setFeedback([[null, null], [null, null]]);
    setShowHint(false);
    setCompleted(false);
  };

  // Get the region of input under kernel for current cell
  const getInputRegion = (row, col) => {
    const region = [];
    for (let i = 0; i < 3; i++) {
      const rowData = [];
      for (let j = 0; j < 3; j++) {
        rowData.push(input[row + i][col + j]);
      }
      region.push(rowData);
    }
    return region;
  };

  const computeHint = (row, col) => {
    const region = getInputRegion(row, col);
    const products = [];
    let sum = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const prod = region[i][j] * kernel[i][j];
        products.push(`${region[i][j]}×${kernel[i][j]}=${prod}`);
        sum += prod;
      }
    }
    return { products, sum };
  };

  const isUnderKernel = (row, col, targetRow, targetCol) => {
    return row >= targetRow && row < targetRow + 3 && col >= targetCol && col < targetCol + 3;
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Interactive Practice</h2>

      {/* Problem display */}
      <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <p className="text-center text-gray-700 mb-3 font-semibold">
          Compute the 2D convolution output
        </p>

        <div className="flex flex-wrap items-start justify-center gap-4">
          {/* Input with kernel highlight */}
          <div>
            <p className="text-sm font-bold text-blue-700 text-center mb-1">Input (4×4)</p>
            <div className="relative">
              <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                {input.flat().map((val, idx) => {
                  const row = Math.floor(idx / 4);
                  const col = idx % 4;
                  const under = isUnderKernel(row, col, currentCell.row, currentCell.col);

                  return (
                    <div
                      key={idx}
                      className={`w-9 h-9 flex items-center justify-center text-sm font-mono font-bold border-2 rounded
                        ${under ? 'bg-purple-100 border-purple-400' : 'bg-blue-50 border-blue-300'}
                      `}
                    >
                      {val}
                    </div>
                  );
                })}
              </div>
              {/* Kernel position indicator */}
              <div
                className="absolute border-3 border-purple-600 rounded pointer-events-none"
                style={{
                  top: currentCell.row * 38 - 1,
                  left: currentCell.col * 38 - 1,
                  width: 3 * 38 + 2,
                  height: 3 * 38 + 2,
                  borderWidth: 3,
                }}
              />
            </div>
          </div>

          {/* Kernel */}
          <div>
            <p className="text-sm font-bold text-green-700 text-center mb-1">Kernel (3×3)</p>
            <p className="text-xs text-gray-500 text-center mb-1">{kernelName}</p>
            <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
              {kernel.flat().map((val, idx) => (
                <div
                  key={idx}
                  className={`w-9 h-9 flex items-center justify-center text-sm font-mono font-bold border-2 rounded
                    ${val > 0 ? 'bg-green-100 border-green-400' : val < 0 ? 'bg-red-100 border-red-400' : 'bg-gray-100 border-gray-300'}
                    ${val < 0 ? 'text-red-700' : 'text-gray-800'}
                  `}
                >
                  {val}
                </div>
              ))}
            </div>
          </div>

          {/* Output */}
          <div>
            <p className="text-sm font-bold text-purple-700 text-center mb-1">Output (2×2)</p>
            <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
              {output.flat().map((_, idx) => {
                const row = Math.floor(idx / 2);
                const col = idx % 2;
                const isCurrent = row === currentCell.row && col === currentCell.col;
                const isCorrect = feedback[row][col];

                return (
                  <div key={idx} className="relative">
                    <input
                      type="number"
                      value={userAnswers[row][col]}
                      onChange={(e) => handleAnswerChange(row, col, e.target.value)}
                      className={`w-12 h-12 text-center text-sm font-mono font-bold border-2 rounded
                        ${isCurrent ? 'border-purple-600 ring-2 ring-purple-300' : 'border-purple-300'}
                        ${isCorrect === true ? 'bg-green-100' : isCorrect === false ? 'bg-red-100' : 'bg-purple-50'}
                        focus:outline-none focus:ring-2 focus:ring-purple-500
                      `}
                      onFocus={() => setCurrentCell({ row, col })}
                    />
                    {isCorrect === true && (
                      <span className="absolute -top-1 -right-1 text-green-600 text-xs">✓</span>
                    )}
                    {isCorrect === false && (
                      <span className="absolute -top-1 -right-1 text-red-600 text-xs">✗</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Current cell calculation */}
      <div className="bg-purple-50 p-3 rounded-lg border border-purple-200 mb-4">
        <div className="flex justify-between items-center mb-2">
          <p className="font-semibold text-purple-700">
            Computing Output[{currentCell.row}, {currentCell.col}]
          </p>
          <button
            onClick={() => setShowHint(!showHint)}
            className="text-sm text-purple-600 hover:underline"
          >
            {showHint ? 'Hide Hint' : 'Show Hint'}
          </button>
        </div>

        {showHint && (
          <div className="text-sm text-gray-600 bg-white p-2 rounded border">
            <p className="mb-1">Element-wise multiply and sum:</p>
            <div className="flex flex-wrap gap-1 font-mono text-xs">
              {computeHint(currentCell.row, currentCell.col).products.map((p, i) => (
                <span key={i} className="bg-gray-100 px-1 rounded">
                  {p}{i < 8 ? ' +' : ''}
                </span>
              ))}
              <span className="font-bold">= {computeHint(currentCell.row, currentCell.col).sum}</span>
            </div>
          </div>
        )}

        <div className="mt-2 flex gap-2">
          <button
            onClick={() => checkAnswer(currentCell.row, currentCell.col)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 text-sm"
          >
            Check Cell
          </button>
          <button
            onClick={checkAll}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
          >
            Check All
          </button>
        </div>
      </div>

      {/* Success */}
      {completed && (
        <div className="p-4 bg-green-100 rounded-lg border border-green-300 text-center mb-4">
          <p className="text-green-700 font-bold text-lg mb-2">🎉 Excellent!</p>
          <p className="text-green-600 mb-3">
            You correctly computed the 2D convolution!
          </p>
          <button
            onClick={newProblem}
            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Try Another Problem
          </button>
        </div>
      )}

      {!completed && (
        <button
          onClick={newProblem}
          className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
        >
          Skip / New Problem
        </button>
      )}

      {/* Info */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
        <p className="text-sm text-gray-700">
          <strong>Common Kernels:</strong>
        </p>
        <ul className="text-xs text-gray-600 mt-1 space-y-1">
          <li><strong>Edge Detection:</strong> Highlights boundaries (positive on one side, negative on other)</li>
          <li><strong>Sharpen:</strong> Enhances edges while keeping center (large positive center, negative surroundings)</li>
          <li><strong>Blur:</strong> Averages neighboring pixels (all same values)</li>
          <li><strong>Identity:</strong> Keeps image unchanged (1 in center, 0 elsewhere)</li>
        </ul>
      </div>
    </div>
  );
}
