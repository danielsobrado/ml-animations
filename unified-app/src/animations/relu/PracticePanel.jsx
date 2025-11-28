import React, { useState, useEffect } from 'react';

// Generate random practice problems
const generateProblem = () => {
  const x = [
    Math.floor(Math.random() * 5) + 1,
    Math.floor(Math.random() * 5) - 2,
    Math.floor(Math.random() * 5) + 1
  ];
  const w = [
    Math.floor(Math.random() * 3) + 1,
    Math.floor(Math.random() * 5) - 2,
    Math.floor(Math.random() * 3) - 1
  ];
  const b = Math.floor(Math.random() * 10) - 5;
  
  const dotProduct = x[0] * w[0] + x[1] * w[1] + x[2] * w[2];
  const z = dotProduct + b;
  const reluOutput = Math.max(0, z);
  
  return { x, w, b, dotProduct, z, reluOutput };
};

const PRACTICE_STEPS = [
  { id: 'dot', label: 'Step 1: Calculate W·X (dot product)' },
  { id: 'z', label: 'Step 2: Calculate z = W·X + b' },
  { id: 'relu', label: 'Step 3: Calculate ReLU(z) = max(0, z)' },
];

export default function PracticePanel({ onStepChange }) {
  const [problem, setProblem] = useState(generateProblem);
  const [currentStep, setCurrentStep] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [feedback, setFeedback] = useState('');
  const [showHint, setShowHint] = useState(false);
  const [completedAnswers, setCompletedAnswers] = useState([null, null, null]);
  const [isComplete, setIsComplete] = useState(false);
  const [score, setScore] = useState(0);
  const [attempts, setAttempts] = useState(0);

  // Notify parent of step changes for graph synchronization
  useEffect(() => {
    if (onStepChange) {
      const completedSteps = completedAnswers.filter(a => a !== null).length;
      onStepChange(completedSteps, problem.z, problem.reluOutput);
    }
  }, [completedAnswers, problem, onStepChange]);

  const getCorrectAnswer = () => {
    switch(currentStep) {
      case 0: return problem.dotProduct;
      case 1: return problem.z;
      case 2: return problem.reluOutput;
      default: return 0;
    }
  };

  const getHint = () => {
    const { x, w, b, dotProduct } = problem;
    switch(currentStep) {
      case 0: 
        return `W·X = (${w[0]}×${x[0]}) + (${w[1]}×${x[1]}) + (${w[2]}×${x[2]})`;
      case 1: 
        return `z = ${dotProduct} + (${b}) = ?`;
      case 2: 
        return `ReLU(${problem.z}) = max(0, ${problem.z}) = ?`;
      default: return '';
    }
  };

  const handleSubmit = () => {
    const userAnswer = parseInt(userInput, 10);
    const correctAnswer = getCorrectAnswer();
    setAttempts(prev => prev + 1);
    
    if (userAnswer === correctAnswer) {
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

  const handleNewProblem = () => {
    setProblem(generateProblem());
    setCurrentStep(0);
    setUserInput('');
    setFeedback('');
    setShowHint(false);
    setCompletedAnswers([null, null, null]);
    setIsComplete(false);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setUserInput('');
    setFeedback('');
    setShowHint(false);
    setCompletedAnswers([null, null, null]);
    setIsComplete(false);
    setScore(0);
    setAttempts(0);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && userInput.trim() !== '') {
      handleSubmit();
    }
  };

  const { x, w, b } = problem;

  return (
    <div className="flex flex-col items-center p-3 h-full">
      <h2 className="text-xl font-bold text-gray-800 mb-2">Practice Exercise</h2>
      
      {/* Problem Display */}
      <div className="bg-white rounded-lg shadow-lg p-4 w-full">
        <div className="flex flex-col items-center gap-4">
          {/* Input Vector X */}
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold w-8">X =</span>
            <div className="flex gap-1">
              {x.map((val, i) => (
                <div
                  key={`x-${i}`}
                  className="w-10 h-10 flex items-center justify-center font-bold text-black rounded bg-blue-400"
                >
                  {val}
                </div>
              ))}
            </div>
          </div>

          {/* Weights Vector W */}
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold w-8">W =</span>
            <div className="flex gap-1">
              {w.map((val, i) => (
                <div
                  key={`w-${i}`}
                  className="w-10 h-10 flex items-center justify-center font-bold text-black rounded bg-green-400"
                >
                  {val}
                </div>
              ))}
            </div>
          </div>

          {/* Bias */}
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold w-8">b =</span>
            <div className="w-10 h-10 flex items-center justify-center font-bold text-black rounded bg-purple-400">
              {b}
            </div>
          </div>
        </div>

        {/* Calculation Flow */}
        <div className="mt-4 flex items-center justify-center gap-2 text-sm flex-wrap">
          <div className={`px-3 py-2 rounded font-bold ${
            completedAnswers[0] !== null ? 'bg-orange-400 text-black' : 'bg-gray-200 text-gray-500'
          }`}>
            W·X = {completedAnswers[0] ?? '?'}
          </div>
          <span className="text-xl">+</span>
          <div className="px-3 py-2 rounded font-bold bg-purple-400 text-black">
            b = {b}
          </div>
          <span className="text-xl">=</span>
          <div className={`px-3 py-2 rounded font-bold ${
            completedAnswers[1] !== null ? 'bg-orange-400 text-black' : 'bg-gray-200 text-gray-500'
          }`}>
            z = {completedAnswers[1] ?? '?'}
          </div>
          <span className="text-xl">→</span>
          <div className={`px-3 py-2 rounded font-bold ${
            completedAnswers[2] !== null ? 'bg-yellow-400 text-black' : 'bg-gray-200 text-gray-500'
          }`}>
            ReLU = {completedAnswers[2] ?? '?'}
          </div>
        </div>

        {/* Current Step Info */}
        <div className="mt-4 text-center">
          <p className="text-gray-700 font-medium">
            {PRACTICE_STEPS[currentStep]?.label || 'Complete!'}
          </p>
        </div>
      </div>

      {/* ReLU Formula Reference */}
      <div className="mt-2 p-2 bg-yellow-100 rounded-lg w-full text-center border border-yellow-300">
        <p className="text-yellow-800 text-sm font-medium">
          ReLU(x) = max(0, x) = { '{' } x if x {'>'} 0, else 0 { '}' }
        </p>
      </div>

      {/* Input Area */}
      {!isComplete ? (
        <div className="mt-3 w-full max-w-sm">
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
              <p className="text-yellow-800 text-sm font-mono">{getHint()}</p>
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
        <div className="mt-3 w-full max-w-sm text-center">
          <div className="p-4 bg-green-100 rounded-lg border border-green-300">
            <p className="text-green-700 font-bold text-lg">🎉 Congratulations!</p>
            <p className="text-green-600 mt-2">
              Score: {score} / {PRACTICE_STEPS.length} correct
            </p>
            <p className="text-green-600 text-sm">
              Total attempts: {attempts}
            </p>
          </div>
          <button
            onClick={handleNewProblem}
            className="mt-3 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-lg transition-colors"
          >
            🎲 New Problem
          </button>
        </div>
      )}

      {/* Progress & Reset */}
      <div className="mt-3 flex items-center gap-4">
        <div className="text-sm text-gray-600">
          Progress: {completedAnswers.filter(a => a !== null).length} / {PRACTICE_STEPS.length}
        </div>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm"
        >
          ↺ Reset
        </button>
        {!isComplete && (
          <button
            onClick={handleNewProblem}
            className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg transition-colors text-sm"
          >
            🎲 New
          </button>
        )}
      </div>
    </div>
  );
}
