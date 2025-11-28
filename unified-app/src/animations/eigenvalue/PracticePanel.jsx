import React, { useState } from 'react';

const generateProblem = () => {
    const problems = [
        { A: [[2, 0], [0, 3]], lambda1: 3, lambda2: 2 },
        { A: [[4, 0], [0, 1]], lambda1: 4, lambda2: 1 },
        { A: [[5, 0], [0, 2]], lambda1: 5, lambda2: 2 },
    ];
    return problems[Math.floor(Math.random() * problems.length)];
};

const PRACTICE_STEPS = [
    { id: 'lambda1', label: 'Step 1: Find largest eigenvalue λ₁' },
    { id: 'lambda2', label: 'Step 2: Find second eigenvalue λ₂' },
];

export default function PracticePanel() {
    const [problem, setProblem] = useState(generateProblem);
    const [currentStep, setCurrentStep] = useState(0);
    const [userInput, setUserInput] = useState('');
    const [feedback, setFeedback] = useState('');
    const [showHint, setShowHint] = useState(false);
    const [completedAnswers, setCompletedAnswers] = useState([null, null]);
    const [isComplete, setIsComplete] = useState(false);
    const [score, setScore] = useState(0);
    const [attempts, setAttempts] = useState(0);

    const getCorrectAnswer = () => {
        return currentStep === 0 ? problem.lambda1 : problem.lambda2;
    };

    const getHint = () => {
        return currentStep === 0
            ? `For diagonal matrix, eigenvalues are the diagonal elements. Take the larger one.`
            : `Take the smaller diagonal element.`;
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
                    setFeedback('🎉 Great! You found all eigenvalues!');
                }
            }, 1000);
        } else {
            setFeedback('✗ Not quite. Try again or ask for a hint.');
        }
    };

    const handleHint = () => setShowHint(true);

    const handleNewProblem = () => {
        setProblem(generateProblem());
        setCurrentStep(0);
        setUserInput('');
        setFeedback('');
        setShowHint(false);
        setCompletedAnswers([null, null]);
        setIsComplete(false);
    };

    const handleReset = () => {
        setCurrentStep(0);
        setUserInput('');
        setFeedback('');
        setShowHint(false);
        setCompletedAnswers([null, null]);
        setIsComplete(false);
        setScore(0);
        setAttempts(0);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && userInput.trim() !== '') handleSubmit();
    };

    const { A } = problem;

    return (
        <div className="flex flex-col items-center p-3 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Practice Exercise</h2>

            <div className="bg-white rounded-lg shadow-lg p-4 w-full">
                <div className="flex flex-col items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-lg font-bold">A =</span>
                        <div className="grid grid-cols-2 gap-1">
                            {A.map((row, i) => row.map((val, j) => (
                                <div key={`a-${i}-${j}`} className="w-12 h-12 flex items-center justify-center font-bold text-black rounded bg-orange-400">
                                    {val}
                                </div>
                            )))}
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <div className={`px-3 py-2 rounded font-bold ${completedAnswers[0] !== null ? 'bg-green-400 text-black' : 'bg-gray-200 text-gray-500'}`}>
                            λ₁ = {completedAnswers[0] ?? '?'}
                        </div>
                        <div className={`px-3 py-2 rounded font-bold ${completedAnswers[1] !== null ? 'bg-green-400 text-black' : 'bg-gray-200 text-gray-500'}`}>
                            λ₂ = {completedAnswers[1] ?? '?'}
                        </div>
                    </div>

                    <div className="mt-2 text-center">
                        <p className="text-gray-700 font-medium">{PRACTICE_STEPS[currentStep]?.label || 'Complete!'}</p>
                    </div>
                </div>
            </div>

            <div className="mt-2 p-2 bg-blue-100 rounded-lg w-full text-center border border-blue-300">
                <p className="text-blue-800 text-xs font-medium">
                    For diagonal matrices, eigenvalues = diagonal elements
                </p>
            </div>

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
                        <button onClick={handleSubmit} disabled={userInput.trim() === ''} className="px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-bold rounded-lg transition-colors">
                            Submit
                        </button>
                    </div>

                    <button onClick={handleHint} className="mt-2 w-full px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white font-bold rounded-lg transition-colors">
                        💡 Show Hint
                    </button>

                    {showHint && (
                        <div className="mt-2 p-3 bg-yellow-100 rounded-lg border border-yellow-300">
                            <p className="text-yellow-800 text-sm">{getHint()}</p>
                        </div>
                    )}

                    {feedback && (
                        <div className={`mt-2 p-3 rounded-lg text-center font-bold ${feedback.includes('✓') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            {feedback}
                        </div>
                    )}
                </div>
            ) : (
                <div className="mt-3 w-full max-w-sm text-center">
                    <div className="p-4 bg-green-100 rounded-lg border border-green-300">
                        <p className="text-green-700 font-bold text-lg">🎉 Congratulations!</p>
                        <p className="text-green-600 mt-2">Score: {score} / {PRACTICE_STEPS.length} correct</p>
                        <p className="text-green-600 text-sm">Total attempts: {attempts}</p>
                    </div>
                    <button onClick={handleNewProblem} className="mt-3 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-lg transition-colors">
                        🎲 New Problem
                    </button>
                </div>
            )}

            <div className="mt-3 flex items-center gap-4">
                <div className="text-sm text-gray-600">Progress: {completedAnswers.filter(a => a !== null).length} / {PRACTICE_STEPS.length}</div>
                <button onClick={handleReset} className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg transition-colors text-sm">↺ Reset</button>
                {!isComplete && <button onClick={handleNewProblem} className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg transition-colors text-sm">🎲 New</button>}
            </div>

            <div className="mt-3 p-3 bg-purple-50 rounded-lg w-full border border-purple-200">
                <p className="text-xs text-purple-800 text-center"><strong>Use in ML:</strong> PCA, Covariance Analysis, Graph Algorithms</p>
            </div>
        </div>
    );
}
