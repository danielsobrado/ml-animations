import React, { useState, useEffect } from 'react';

const ALPHA = 0.01;

function generateProblem() {
  // Generate random X (3 values between -3 and 5)
  const X = Array.from({ length: 3 }, () => Math.floor(Math.random() * 9) - 3);
  // Generate random W (3 values between -2 and 2)
  const W = Array.from({ length: 3 }, () => Math.floor(Math.random() * 5) - 2);
  // Generate random b (between -8 and 4)
  const b = Math.floor(Math.random() * 13) - 8;
  
  const dotProduct = X.reduce((sum, x, i) => sum + x * W[i], 0);
  const z = dotProduct + b;
  const leakyRelu = z > 0 ? z : ALPHA * z;
  
  return { X, W, b, dotProduct, z, leakyRelu };
}

export default function PracticePanel({ onStepChange }) {
  const [problem, setProblem] = useState(generateProblem);
  const [currentStep, setCurrentStep] = useState(1);
  const [userAnswers, setUserAnswers] = useState({
    dotProduct: '',
    z: '',
    leakyRelu: ''
  });
  const [feedback, setFeedback] = useState({
    dotProduct: null,
    z: null,
    leakyRelu: null
  });
  const [showHints, setShowHints] = useState({
    dotProduct: false,
    z: false,
    leakyRelu: false
  });

  useEffect(() => {
    if (onStepChange) {
      onStepChange(currentStep, problem.z, problem.leakyRelu);
    }
  }, [currentStep, problem.z, problem.leakyRelu, onStepChange]);

  const checkAnswer = (field) => {
    const userVal = parseFloat(userAnswers[field]);
    let correct;
    
    if (field === 'dotProduct') {
      correct = userVal === problem.dotProduct;
    } else if (field === 'z') {
      correct = userVal === problem.z;
    } else {
      // For leaky relu, allow some floating point tolerance
      correct = Math.abs(userVal - problem.leakyRelu) < 0.001;
    }
    
    setFeedback(prev => ({ ...prev, [field]: correct }));
    
    if (correct) {
      if (field === 'dotProduct') setCurrentStep(2);
      else if (field === 'z') setCurrentStep(3);
      else setCurrentStep(4);
    }
  };

  const newProblem = () => {
    setProblem(generateProblem());
    setCurrentStep(1);
    setUserAnswers({ dotProduct: '', z: '', leakyRelu: '' });
    setFeedback({ dotProduct: null, z: null, leakyRelu: null });
    setShowHints({ dotProduct: false, z: false, leakyRelu: false });
  };

  const toggleHint = (field) => {
    setShowHints(prev => ({ ...prev, [field]: !prev[field] }));
  };

  const { X, W, b, dotProduct, z, leakyRelu } = problem;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold text-gray-800 mb-3 text-center">Interactive Practice</h2>
      
      {/* Problem Display */}
      <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-sm font-semibold text-blue-600 mb-1">X (Input)</p>
            <p className="text-lg font-mono">[{X.join(', ')}]</p>
          </div>
          <div>
            <p className="text-sm font-semibold text-green-600 mb-1">W (Weights)</p>
            <p className="text-lg font-mono">[{W.join(', ')}]</p>
          </div>
          <div>
            <p className="text-sm font-semibold text-yellow-600 mb-1">b (Bias)</p>
            <p className="text-lg font-mono">{b}</p>
          </div>
        </div>
        <div className="mt-2 text-center">
          <p className="text-sm font-semibold text-orange-600">α (Alpha)</p>
          <p className="text-lg font-mono">{ALPHA}</p>
        </div>
      </div>

      {/* Step 1: Dot Product */}
      <div className={`p-4 rounded-lg border mb-3 ${currentStep >= 1 ? 'bg-white border-purple-300' : 'bg-gray-100 border-gray-200'}`}>
        <div className="flex justify-between items-center mb-2">
          <p className="font-semibold text-purple-700">Step 1: Calculate X · W</p>
          <button
            onClick={() => toggleHint('dotProduct')}
            className="text-sm text-purple-600 hover:underline"
          >
            {showHints.dotProduct ? 'Hide Hint' : 'Show Hint'}
          </button>
        </div>
        
        {showHints.dotProduct && (
          <p className="text-sm text-gray-600 mb-2">
            X · W = ({X[0]} × {W[0]}) + ({X[1]} × {W[1]}) + ({X[2]} × {W[2]})
            = {X[0] * W[0]} + {X[1] * W[1]} + {X[2] * W[2]} = ?
          </p>
        )}
        
        <div className="flex gap-2 items-center">
          <input
            type="number"
            value={userAnswers.dotProduct}
            onChange={(e) => setUserAnswers(prev => ({ ...prev, dotProduct: e.target.value }))}
            disabled={feedback.dotProduct === true}
            className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            placeholder="Enter dot product"
          />
          <button
            onClick={() => checkAnswer('dotProduct')}
            disabled={feedback.dotProduct === true}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
          >
            Check
          </button>
        </div>
        {feedback.dotProduct !== null && (
          <p className={`mt-2 text-sm ${feedback.dotProduct ? 'text-green-600' : 'text-red-600'}`}>
            {feedback.dotProduct ? '✓ Correct!' : `✗ Try again. Hint: ${X[0]}×${W[0]} + ${X[1]}×${W[1]} + ${X[2]}×${W[2]}`}
          </p>
        )}
      </div>

      {/* Step 2: z = dot + b */}
      <div className={`p-4 rounded-lg border mb-3 ${currentStep >= 2 ? 'bg-white border-red-300' : 'bg-gray-100 border-gray-200'}`}>
        <div className="flex justify-between items-center mb-2">
          <p className="font-semibold text-red-700">Step 2: Calculate z = X·W + b</p>
          {currentStep >= 2 && (
            <button
              onClick={() => toggleHint('z')}
              className="text-sm text-red-600 hover:underline"
            >
              {showHints.z ? 'Hide Hint' : 'Show Hint'}
            </button>
          )}
        </div>
        
        {showHints.z && currentStep >= 2 && (
          <p className="text-sm text-gray-600 mb-2">
            z = {dotProduct} + ({b}) = ?
          </p>
        )}
        
        <div className="flex gap-2 items-center">
          <input
            type="number"
            value={userAnswers.z}
            onChange={(e) => setUserAnswers(prev => ({ ...prev, z: e.target.value }))}
            disabled={currentStep < 2 || feedback.z === true}
            className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 disabled:bg-gray-100"
            placeholder="Enter z value"
          />
          <button
            onClick={() => checkAnswer('z')}
            disabled={currentStep < 2 || feedback.z === true}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
          >
            Check
          </button>
        </div>
        {feedback.z !== null && (
          <p className={`mt-2 text-sm ${feedback.z ? 'text-green-600' : 'text-red-600'}`}>
            {feedback.z ? '✓ Correct!' : `✗ Try again. z = ${dotProduct} + (${b})`}
          </p>
        )}
      </div>

      {/* Step 3: Leaky ReLU */}
      <div className={`p-4 rounded-lg border mb-3 ${currentStep >= 3 ? 'bg-white border-orange-300' : 'bg-gray-100 border-gray-200'}`}>
        <div className="flex justify-between items-center mb-2">
          <p className="font-semibold text-orange-700">Step 3: Apply Leaky ReLU</p>
          {currentStep >= 3 && (
            <button
              onClick={() => toggleHint('leakyRelu')}
              className="text-sm text-orange-600 hover:underline"
            >
              {showHints.leakyRelu ? 'Hide Hint' : 'Show Hint'}
            </button>
          )}
        </div>
        
        {showHints.leakyRelu && currentStep >= 3 && (
          <p className="text-sm text-gray-600 mb-2">
            {z > 0 
              ? `z = ${z} > 0, so Leaky ReLU(${z}) = ${z}` 
              : `z = ${z} ≤ 0, so Leaky ReLU(${z}) = α × ${z} = ${ALPHA} × ${z} = ?`}
          </p>
        )}
        
        <div className="flex gap-2 items-center">
          <input
            type="number"
            step="0.01"
            value={userAnswers.leakyRelu}
            onChange={(e) => setUserAnswers(prev => ({ ...prev, leakyRelu: e.target.value }))}
            disabled={currentStep < 3 || feedback.leakyRelu === true}
            className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 disabled:bg-gray-100"
            placeholder="Enter Leaky ReLU output"
          />
          <button
            onClick={() => checkAnswer('leakyRelu')}
            disabled={currentStep < 3 || feedback.leakyRelu === true}
            className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50"
          >
            Check
          </button>
        </div>
        {feedback.leakyRelu !== null && (
          <p className={`mt-2 text-sm ${feedback.leakyRelu ? 'text-green-600' : 'text-red-600'}`}>
            {feedback.leakyRelu 
              ? '✓ Correct! 🎉' 
              : `✗ Try again. ${z > 0 ? `z > 0, so output = ${z}` : `z ≤ 0, so output = ${ALPHA} × ${z}`}`}
          </p>
        )}
      </div>

      {/* Success / New Problem */}
      {currentStep === 4 && (
        <div className="p-4 bg-green-100 rounded-lg border border-green-300 text-center">
          <p className="text-green-700 font-bold text-lg mb-2">🎉 Excellent Work!</p>
          <p className="text-green-600 mb-3">
            You correctly computed Leaky ReLU({z}) = {leakyRelu.toFixed(2)}
          </p>
          <button
            onClick={newProblem}
            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
          >
            Try Another Problem
          </button>
        </div>
      )}

      {currentStep < 4 && (
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
