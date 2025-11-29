import React, { useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { steps } from './stepsConfig';
import Step1Tokenization from './steps/Step1Tokenization';
import Step2Positional from './steps/Step2Positional';
import Step3Attention from './steps/Step3Attention';
import Step4FFN from './steps/Step4FFN';
import Step5Norm from './steps/Step5Norm';
import Step6Architecture from './steps/Step6Architecture';
import Step7WeightTying from './steps/Step7WeightTying';
import Step8Training from './steps/Step8Training';
import Step9Inference from './steps/Step9Inference';

export default function App() {
    const [completedSteps, setCompletedSteps] = useState(new Set());
    const navigate = useNavigate();
    const location = useLocation();

    const currentStepId = steps.find(s => s.path === location.pathname)?.id || 1;

    const markComplete = (stepId) => {
        setCompletedSteps(prev => new Set([...prev, stepId]));
    };

    const goToNext = () => {
        const currentIndex = steps.findIndex(s => s.id === currentStepId);
        if (currentIndex < steps.length - 1) {
            navigate(steps[currentIndex + 1].path);
        }
    };

    const goToPrev = () => {
        const currentIndex = steps.findIndex(s => s.id === currentStepId);
        if (currentIndex > 0) {
            navigate(steps[currentIndex - 1].path);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    <h1 className="text-2xl font-bold text-emerald-400">GPT-2 Deep Dive</h1>
                    <div className="text-sm text-gray-400">
                        Step {currentStepId} of {steps.length}
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex">
                {/* Sidebar */}
                <aside className="w-64 bg-gray-800 min-h-[calc(100vh-73px)] border-r border-gray-700 p-4">
                    <nav>
                        <ul className="space-y-2">
                            {steps.map((step) => (
                                <li key={step.id}>
                                    <button
                                        onClick={() => navigate(step.path)}
                                        className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${currentStepId === step.id
                                                ? 'bg-emerald-600 text-white'
                                                : completedSteps.has(step.id)
                                                    ? 'bg-gray-700 text-emerald-400'
                                                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                            }`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="font-mono text-xs">{step.id}</span>
                                            <div className="flex-1">
                                                <div className="font-semibold text-sm">{step.title}</div>
                                                <div className="text-xs opacity-75">{step.description}</div>
                                            </div>
                                            {completedSteps.has(step.id) && (
                                                <span className="text-emerald-400">✓</span>
                                            )}
                                        </div>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </nav>
                </aside>

                {/* Content */}
                <main className="flex-1 p-8">
                    <div className="max-w-4xl mx-auto">
                        <Routes>
                            <Route path="/" element={<Step1Tokenization onComplete={() => markComplete(1)} onNext={goToNext} />} />
                            <Route path="/step1" element={<Step1Tokenization onComplete={() => markComplete(1)} onNext={goToNext} />} />
                            <Route path="/step2" element={<Step2Positional onComplete={() => markComplete(2)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step3" element={<Step3Attention onComplete={() => markComplete(3)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step4" element={<Step4FFN onComplete={() => markComplete(4)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step5" element={<Step5Norm onComplete={() => markComplete(5)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step6" element={<Step6Architecture onComplete={() => markComplete(6)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step7" element={<Step7WeightTying onComplete={() => markComplete(7)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step8" element={<Step8Training onComplete={() => markComplete(8)} onNext={goToNext} onPrev={goToPrev} />} />
                            <Route path="/step9" element={<Step9Inference onComplete={() => markComplete(9)} onPrev={goToPrev} />} />
                        </Routes>
                    </div>
                </main>
            </div>
        </div>
    );
}
