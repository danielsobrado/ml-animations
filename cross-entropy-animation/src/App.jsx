import React, { useState } from 'react';
import CrossEntropyPanel from './CrossEntropyPanel';
import LogLossGraphPanel from './LogLossGraphPanel';
import PracticePanel from './PracticePanel';

export default function App() {
    const [animProb, setAnimProb] = useState(0.7);
    const [practiceProb, setPracticeProb] = useState(0.7);
    const [isAnimating, setIsAnimating] = useState(true);

    const handleAnimationStepChange = (step) => {
        setIsAnimating(true);
        // In animation, we just use a fixed example p=0.7 for simplicity in the graph
        // unless we want to animate p changing, but the animation is about steps.
        // Let's keep p=0.7 for the animation context.
        setAnimProb(0.7);
    };

    const handlePracticeChange = (prob) => {
        setPracticeProb(prob);
        setIsAnimating(false);
    };

    const currentProb = isAnimating ? animProb : practiceProb;

    return (
        <div className="min-h-screen bg-gray-100 p-4">
            <h1 className="text-3xl font-bold text-gray-800 text-center mb-4">Cross-Entropy Loss</h1>

            <div className="flex flex-col gap-4 max-w-7xl mx-auto">
                {/* Top Row - Animation and Practice */}
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Left Panel - Animation Demo */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden" onClick={() => setIsAnimating(true)}>
                        <CrossEntropyPanel onStepChange={handleAnimationStepChange} />
                    </div>

                    {/* Right Panel - Interactive Practice */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden" onClick={() => setIsAnimating(false)}>
                        <PracticePanel onStepChange={handlePracticeChange} />
                    </div>
                </div>

                {/* Bottom Panel - Graph */}
                <div className="bg-gray-50 rounded-xl shadow-lg overflow-hidden p-4">
                    <h2 className="text-xl font-bold text-gray-800 text-center mb-3">Visualization</h2>
                    <div className="flex justify-center">
                        <LogLossGraphPanel
                            probability={currentProb}
                            isActive={true}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
