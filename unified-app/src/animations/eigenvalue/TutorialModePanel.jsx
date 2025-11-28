import React, { useState } from 'react';

const TUTORIAL_STEPS = [
    {
        title: "What is a Linear Transformation?",
        content: "A matrix A transforms vectors: it takes an input vector v and produces an output vector Av. Think of it as a function that moves and stretches vectors in space.",
        visual: "grid",
        key: "transform"
    },
    {
        title: "Most Vectors Change Direction",
        content: "When you apply matrix A to most vectors, they change BOTH direction and length. The output vector Av points in a completely different direction from the input v.",
        visual: "random",
        key: "change"
    },
    {
        title: "Special Vectors: Eigenvectors",
        content: "But some special vectors called EIGENVECTORS only change their LENGTH, not their DIRECTION. After transformation, Av points in the same direction as v (or exactly opposite).",
        visual: "eigenvector",
        key: "eigen"
    },
    {
        title: "The Stretch Factor: Eigenvalue",
        content: "The factor by which an eigenvector stretches is called the EIGENVALUE (λ). If v is an eigenvector, then Av = λv. The eigenvalue tells you: 'how much does this special direction stretch?'",
        visual: "eigenvalue",
        key: "lambda"
    },
    {
        title: "The Equation: Av = λv",
        content: "This is the fundamental eigenvalue equation. It says: 'Applying matrix A to eigenvector v just multiplies it by scalar λ.' No rotation, only scaling!",
        visual: "equation",
        key: "equation"
    },
    {
        title: "Geometric Interpretation",
        content: "When matrix A transforms a circle into an ellipse, the ellipse's axes ARE the eigenvector directions! The axis lengths ARE the eigenvalues!",
        visual: "ellipse",
        key: "geometric"
    },
    {
        title: "Eigenvalue Decomposition",
        content: "For symmetric matrices, A = QΛQ^T where Q contains eigenvectors as columns, and Λ is diagonal with eigenvalues. This decomposition reveals the matrix's fundamental geometry.",
        visual: "decomposition",
        key: "decomp"
    }
];

export default function TutorialModePanel() {
    const [currentStep, setCurrentStep] = useState(0);

    const step = TUTORIAL_STEPS[currentStep];

    const nextStep = () => {
        if (currentStep < TUTORIAL_STEPS.length - 1) {
            setCurrentStep(currentStep + 1);
        }
    };

    const prevStep = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const getVisualComponent = () => {
        switch (step.visual) {
            case "grid":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg">
                        <div className="text-center">
                            <p className="text-6xl mb-2">📐</p>
                            <p className="text-sm text-blue-900 font-bold">Grid Transformation</p>
                            <p className="text-xs text-blue-800 mt-1">Matrix A warps the coordinate grid</p>
                        </div>
                    </div>
                );

            case "random":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-orange-100 to-red-100 rounded-lg">
                        <div className="text-center">
                            <p className="text-6xl mb-2">🔄</p>
                            <p className="text-sm text-red-900 font-bold">Direction + Length Change</p>
                            <p className="text-xs text-red-800 mt-1">v → Av (different direction)</p>
                        </div>
                    </div>
                );

            case "eigenvector":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-green-100 to-green-200 rounded-lg">
                        <div className="text-center">
                            <p className="text-6xl mb-2">➡️</p>
                            <p className="text-sm text-green-900 font-bold">Same Direction!</p>
                            <p className="text-xs text-green-800 mt-1">v → λv (only stretches)</p>
                        </div>
                    </div>
                );

            case "eigenvalue":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-purple-100 to-purple-200 rounded-lg">
                        <div className="text-center">
                            <p className="text-6xl mb-2">×λ</p>
                            <p className="text-sm text-purple-900 font-bold">Eigenvalue = Scale Factor</p>
                            <p className="text-xs text-purple-800 mt-1">||Av|| = λ × ||v||</p>
                        </div>
                    </div>
                );

            case "equation":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-yellow-100 to-yellow-200 rounded-lg">
                        <div className="text-center p-6">
                            <p className="text-4xl font-bold text-yellow-900 mb-3">A v = λ v</p>
                            <div className="text-sm text-yellow-800 space-y-1">
                                <p>A = matrix (transformation)</p>
                                <p>v = eigenvector (special direction)</p>
                                <p>λ = eigenvalue (stretch factor)</p>
                            </div>
                        </div>
                    </div>
                );

            case "ellipse":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-indigo-100 to-indigo-200 rounded-lg">
                        <div className="text-center">
                            <p className="text-6xl mb-2">⭕ → 🥚</p>
                            <p className="text-sm text-indigo-900 font-bold">Circle → Ellipse</p>
                            <p className="text-xs text-indigo-800 mt-1">Ellipse axes = eigenvectors</p>
                            <p className="text-xs text-indigo-800">Axis lengths = eigenvalues</p>
                        </div>
                    </div>
                );

            case "decomposition":
                return (
                    <div className="flex items-center justify-center h-48 bg-gradient-to-br from-pink-100 to-pink-200 rounded-lg">
                        <div className="text-center p-4">
                            <p className="text-3xl font-bold text-pink-900 mb-2">A = Q Λ Q^T</p>
                            <div className="text-xs text-pink-800 space-y-1">
                                <p>Q = eigenvectors (orthonormal)</p>
                                <p>Λ = eigenvalues (diagonal)</p>
                                <p>Reveals fundamental geometry!</p>
                            </div>
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-3">Tutorial: Understanding Eigenvalues</h2>

            <div className="w-full max-w-2xl">
                {/* Progress bar */}
                <div className="mb-4">
                    <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Step {currentStep + 1} of {TUTORIAL_STEPS.length}</span>
                        <span>{Math.round(((currentStep + 1) / TUTORIAL_STEPS.length) * 100)}% Complete</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${((currentStep + 1) / TUTORIAL_STEPS.length) * 100}%` }}
                        />
                    </div>
                </div>

                {/* Content */}
                <div className="bg-white rounded-xl shadow-lg p-6 border-2 border-gray-300">
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">{step.title}</h3>

                    {getVisualComponent()}

                    <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <p className="text-gray-800 leading-relaxed">{step.content}</p>
                    </div>
                </div>

                {/* Navigation */}
                <div className="flex items-center justify-between mt-6">
                    <button
                        onClick={prevStep}
                        disabled={currentStep === 0}
                        className="px-6 py-3 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors"
                    >
                        ← Previous
                    </button>

                    <div className="flex gap-2">
                        {TUTORIAL_STEPS.map((_, idx) => (
                            <button
                                key={idx}
                                onClick={() => setCurrentStep(idx)}
                                className={`w-3 h-3 rounded-full transition-colors ${idx === currentStep ? 'bg-blue-600' : 'bg-gray-300 hover:bg-gray-400'
                                    }`}
                            />
                        ))}
                    </div>

                    <button
                        onClick={nextStep}
                        disabled={currentStep === TUTORIAL_STEPS.length - 1}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold rounded-lg transition-colors"
                    >
                        {currentStep === TUTORIAL_STEPS.length - 1 ? '✓ Done' : 'Next →'}
                    </button>
                </div>

                {/* Key concept callout */}
                <div className="mt-4 p-4 bg-green-50 rounded-lg border-2 border-green-300">
                    <p className="text-sm text-green-900">
                        <strong>🎯 Key Takeaway:</strong> Eigenvalues and eigenvectors reveal the "natural directions"
                        of a matrix transformation—the directions that only stretch, never rotate!
                    </p>
                </div>
            </div>
        </div>
    );
}
