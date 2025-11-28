import React, { useState } from 'react';

// Different matrices for practice (different from the animation example)
const practiceMatrixA = [[1, 2], [3, 1]];
const practiceMatrixB = [[2, 1, 3], [1, 4, 2]];
const practiceResult = [[4, 9, 7], [7, 7, 11]];

const PRACTICE_STEPS = [
  { row: 0, col: 0, hint: 'Multiply Row 1 of A with Column 1 of B: (1×2) + (2×1)', answer: 4 },
  { row: 0, col: 1, hint: 'Multiply Row 1 of A with Column 2 of B: (1×1) + (2×4)', answer: 9 },
  { row: 0, col: 2, hint: 'Multiply Row 1 of A with Column 3 of B: (1×3) + (2×2)', answer: 7 },
  { row: 1, col: 0, hint: 'Multiply Row 2 of A with Column 1 of B: (3×2) + (1×1)', answer: 7 },
  { row: 1, col: 1, hint: 'Multiply Row 2 of A with Column 2 of B: (3×1) + (1×4)', answer: 7 },
  { row: 1, col: 2, hint: 'Multiply Row 2 of A with Column 3 of B: (3×3) + (1×2)', answer: 11 },
];

export default function PracticePanel() {
  const [currentStep, setCurrentStep] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [showHint, setShowHint] = useState(false);
  const [completedAnswers, setCompletedAnswers] = useState(Array(6).fill(null));
  const [isComplete, setIsComplete] = useState(false);
  const [score, setScore] = useState(0);
  const [attempts, setAttempts] = useState(0);

  const currentPractice = PRACTICE_STEPS[currentStep];

  const handleSubmit = () => {
    const userAnswer = parseInt(userInput, 10);
    setAttempts(prev => prev + 1);
    
    if (userAnswer === currentPractice.answer) {
      setFeedback('✓ Correct!');
      setScore(prev => prev + 1);
      
      const newAnswers = [...completedAnswers];
      newAnswers[currentStep] = userAnswer;
      setCompletedAnswers(newAnswers);
      
      setTimeout(() => {
        if (currentStep < PRACTICE_STEPS.length - 1) {
          setCurrentStep(prev => prev + 1);
          setUserInput('');
          setFeedback('');
          setShowHint(false);
        } else {
          setIsComplete(true);
          setFeedback('🎉 Excellent! You completed all steps!');
        }
      }, 1000);
    } else {
      setFeedback('✗ Not quite. Try again or ask for a hint.');
    }
  };

  const handleHint = () => {
    setShowHint(true);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setUserInput('');
    setFeedback('');
    setShowHint(false);
    setCompletedAnswers(Array(6).fill(null));
    setIsComplete(false);
    setScore(0);
    setAttempts(0);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && userInput.trim() !== '') {
      handleSubmit();
    }
  };

  const getCellColor = (matrixType) => {
    switch(matrixType) {
      case 'A': return 'bg-blue-400';
      case 'B': return 'bg-green-400';
      case 'R': return 'bg-orange-400';
      default: return 'bg-gray-300';
    }
  };

  const isHighlightedRow = (row) => currentStep < PRACTICE_STEPS.length && PRACTICE_STEPS[currentStep].row === row;
  const isHighlightedCol = (col) => currentStep < PRACTICE_STEPS.length && PRACTICE_STEPS[currentStep].col === col;

  return (
    <div className="flex flex-col items-center p-3 h-full">
      <h2 className="text-xl font-bold text-gray-800 mb-2">Practice Exercise</h2>
      
      {/* Matrices Display */}
      <div className="bg-white rounded-lg shadow-lg p-4 w-full">
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {/* Matrix A */}
          <div className="flex flex-col items-center">
            <span className="text-lg font-bold mb-1">A</span>
            <div className="grid grid-cols-2 gap-1">
              {practiceMatrixA.map((row, i) => (
                row.map((val, j) => (
                  <div
                    key={`a-${i}-${j}`}
                    className={`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${
                      isHighlightedRow(i) ? 'bg-blue-300 scale-110 ring-2 ring-blue-500' : 'bg-blue-400'
                    } transition-all`}
                  >
                    {val}
                  </div>
                ))
              ))}
            </div>
          </div>

          <span className="text-2xl font-bold mx-2">×</span>

          {/* Matrix B */}
          <div className="flex flex-col items-center">
            <span className="text-lg font-bold mb-1">B</span>
            <div className="grid grid-cols-3 gap-1">
              {practiceMatrixB.map((row, i) => (
                row.map((val, j) => (
                  <div
                    key={`b-${i}-${j}`}
                    className={`w-10 h-10 flex items-center justify-center font-bold text-black rounded ${
                      isHighlightedCol(j) ? 'bg-green-300 scale-110 ring-2 ring-green-500' : 'bg-green-400'
                    } transition-all`}
                  >
                    {val}
                  </div>
                ))
              ))}
            </div>
          </div>

          <span className="text-2xl font-bold mx-2">=</span>

          {/* Result Matrix */}
          <div className="flex flex-col items-center">
            <span className="text-lg font-bold mb-1">C</span>
            <div className="grid grid-cols-3 gap-1">
              {practiceResult.map((row, i) => (
                row.map((val, j) => {
                  const stepIndex = i * 3 + j;
                  const isCurrentCell = currentStep === stepIndex;
                  const isCompleted = completedAnswers[stepIndex] !== null;
                  
                  return (
                    <div
                      key={`r-${i}-${j}`}
                      className={`w-10 h-10 flex items-center justify-center font-bold text-black rounded transition-all ${
                        isCurrentCell 
                          ? 'bg-yellow-300 ring-2 ring-yellow-500 scale-110' 
                          : isCompleted 
                            ? 'bg-orange-400' 
                            : 'bg-orange-200'
                      }`}
                    >
                      {isCompleted ? completedAnswers[stepIndex] : '?'}
                    </div>
                  );
                })
              ))}
            </div>
          </div>
        </div>

        {/* Current Step Info */}
        <div className="mt-4 text-center">
          <p className="text-gray-700 font-medium">
            Step {currentStep + 1} of {PRACTICE_STEPS.length}: Calculate C[{currentPractice.row + 1}][{currentPractice.col + 1}]
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Row {currentPractice.row + 1} of A × Column {currentPractice.col + 1} of B
          </p>
        </div>
      </div>

      {/* Input Area */}
      {!isComplete ? (
        <div className="mt-4 w-full max-w-sm">
          <div className="flex gap-2">
            <input
              type="number"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Your answer..."
              className="flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-center text-lg font-bold"
            />
            <button
              onClick={handleSubmit}
              disabled={userInput.trim() === ''}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors"
            >
              Submit
            </button>
          </div>
          
          <button
            onClick={handleHint}
            className="mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors"
          >
            💡 Show Hint
          </button>

          {showHint && (
            <div className="mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300">
              <p className="text-yellow-800 text-sm">{currentPractice.hint}</p>
            </div>
          )}

          {feedback && (
            <div className={`mt-2 p-3 rounded-lg text-center font-bold ${
              feedback.includes('✓') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
              {feedback}
            </div>
          )}
        </div>
      ) : (
        <div className="mt-4 w-full max-w-sm text-center">
          <div className="p-4 bg-green-100 rounded-lg border border-green-300">
            <p className="text-green-700 font-bold text-lg">🎉 Congratulations!</p>
            <p className="text-green-600 mt-2">
              Score: {score} / {PRACTICE_STEPS.length} correct
            </p>
            <p className="text-green-600 text-sm">
              Total attempts: {attempts}
            </p>
          </div>
        </div>
      )}

      {/* Progress & Reset */}
      <div className="mt-4 flex items-center gap-4">
        <div className="text-sm text-gray-600">
          Progress: {completedAnswers.filter(a => a !== null).length} / {PRACTICE_STEPS.length}
        </div>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm"
        >
          ↺ Reset
        </button>
      </div>
    </div>
  );
}
