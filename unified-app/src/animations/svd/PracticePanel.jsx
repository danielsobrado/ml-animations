import React, { useState } from 'react';

// Generate simple 2x2 practice problems for SVD
const generateProblem = () => {
    // Use simple matrices where SVD can be computed without numerical libraries
    const problems = [
        {
            A: [[3, 0], [0, 2]],
            sigma1: 3,
            sigma2: 2,
            U: [[1, 0], [0, 1]],
            VT: [[1, 0], [0, 1]]
        },
        {
            A: [[2, 0], [0, 1]],
            sigma1: 2,
            sigma2: 1,
            U: [[1, 0], [0, 1]],
            VT: [[1, 0], [0, 1]]
        },
        {
            A: [[4, 0], [0, 3]],
            sigma1: 4,
            sigma2: 3,
            U: [[1, 0], [0, 1]],
            VT: [[1, 0], [0, 1]]
        }
    ];

    return problems[Math.floor(Math.random() * problems.length)];
};

const PRACTICE_STEPS = [
    { id: 'sigma1', label: 'Step 1: Find largest singular value σ₁' },
    { id: 'sigma2', label: 'Step 2: Find second singular value σ₂' },
    { id: 'verify', label: 'Step 3: Verify σ₁ ≥ σ₂ ≥ 0' },
];

export default function PracticePanel() {
    const [problem, setProblem] = useState(generateProblem);
    const [currentStep, setCurrentStep] = useState(0);
    const [userInput, setUserInput] = useState('');
    const [feedback, setFeedback] = useState('');
    const [showHint, setShowHint] = useState(false);
    const [completedAnswers, setCompletedAnswers] = useState([null, null, null]);
    const [isComplete, setIsComplete] = useState(false);
    const [score, setScore] = useState(0);
    const [attempts, setAttempts] = useState(0);

    const getCorrectAnswer = () => {
        switch (currentStep) {
            case 0: return problem.sigma1;
            case 1: return problem.sigma2;
            case 2: return 1; // Verification: 1 for true
            default: return 0;
        }
    };

    const getHint = () => {
        switch (currentStep) {
            case 0:
                return `Compute A^T A, then find eigenvalues. σ₁ = √(largest eigenvalue)`;
            case 1:
                return `σ₂ = √(second largest eigenvalue of A^T A)`;
            case 2:
                return `Check: Is ${problem.sigma1} ≥ ${problem.sigma2} ≥ 0? (Enter 1 for Yes, 0 for No)`;
            default: return '';
        }
    };

    const handleSubmit = () => {
        const userAnswer = parseFloat(userInput);
        const correctAnswer = getCorrectAnswer();
        setAttempts(prev => prev + 1);

        if (Math.abs(userAnswer - correctAnswer) < 0.01) {
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
                    setFeedback('🎉 Excellent! You completed all SVD steps!');
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

    const { A } = problem;

    return (
        <div className="flex flex-col items-center p-3 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Practice Exercise</h2>

            {/* Problem Display */}
            <div className="bg-white rounded-lg shadow-lg p-4 w-full">
                <div className="flex flex-col items-center gap-4">
                    {/* Matrix A */}
                    <div className="flex items-center gap-2">
                        <span className="text-lg font-bold">A =</span>
                        <div className="grid grid-cols-2 gap-1">
                            {A.map((row, i) => (
                                row.map((val, j) => (
                                    <div
                                        key={`a-${i}-${j}`}
                                        className="w-12 h-12 flex items-center justify-center font-bold text-black rounded bg-orange-400"
                                    >
                                        {val}
                                    </div>
                                ))
                            ))}
                        </div>
                    </div>

                    {/* SVD Formula */}
                    <div className="text-center text-sm text-gray-700">
                        Find: A = U Σ V^T
                    </div>

                    {/* Singular Values Display */}
                    <div className="flex items-center gap-2 flex-wrap justify-center">
                        <div className={`px-3 py-2 rounded font-bold ${completedAnswers[0] !== null ? 'bg-purple-400 text-black' : 'bg-gray-200 text-gray-500'
                            }`}>
                            σ₁ = {completedAnswers[0] ?? '?'}
                        </div>
                        <span className="text-xl">≥</span>
                        <div className={`px-3 py-2 rounded font-bold ${completedAnswers[1] !== null ? 'bg-purple-400 text-black' : 'bg-gray-200 text-gray-500'
                            }`}>
                            σ₂ = {completedAnswers[1] ?? '?'}
                        </div>
                        <span className="text-xl">≥</span>
                        <div className="px-3 py-2 rounded font-bold bg-purple-400 text-black">
                            0
                        </div>
                    </div>

                    {/* Verification */}
                    {currentStep === 2 && (
                        <div className="text-center">
                            <p className="text-sm text-gray-600">Is the ordering correct?</p>
                        </div>
                    )}

                    {/* Current Step Info */}
                    <div className="mt-2 text-center">
                        <p className="text-gray-700 font-medium">
                            {PRACTICE_STEPS[currentStep]?.label || 'Complete!'}
                        </p>
                    </div>
                </div>
            </div>

            {/* SVD Formula Reference */}
            <div className="mt-2 p-2 bg-purple-100 rounded-lg w-full text-center border border-purple-300">
                <p className="text-purple-800 text-xs font-medium">
                    Hint: For diagonal A, σᵢ are the absolute values of diagonal elements
                </p>
            </div>

            {/* Input Area */}
            {!isComplete ? (
                <div className="mt-3 w-full max-w-sm">
                    <div className="flex gap-2">
                        <input
                            type="number"
                            step="0.01"
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
                        <div className={`mt-2 p-3 rounded-lg text-center font-bold ${feedback.includes('✓') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
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

            {/* Educational Info */}
            <div className="mt-3 p-3 bg-blue-50 rounded-lg w-full border border-blue-200">
                <p className="text-xs text-blue-800 text-center">
                    <strong>SVD Applications in ML:</strong> PCA, Image Compression, Recommender Systems, LSA
                </p>
            </div>
        </div>
    );
}
