import React, { useState } from 'react';
import SoftmaxAnimationPanel from './SoftmaxAnimationPanel';
import SoftmaxGraphPanel from './SoftmaxGraphPanel';
import PracticePanel from './PracticePanel';

export default function App() {
    const [animationLogits, setAnimationLogits] = useState([]);
    const [animationProbs, setAnimationProbs] = useState([]);
    const [practiceLogits, setPracticeLogits] = useState([]);
    const [practiceProbs, setPracticeProbs] = useState([]);
    const [isAnimating, setIsAnimating] = useState(true); // Track which source to show

    const handleAnimationStepChange = (step, logits, probs) => {
        setAnimationLogits(logits);
        setAnimationProbs(probs);
        setIsAnimating(true);
    };

    const handlePracticeChange = (logits, probs) => {
        setPracticeLogits(logits);
        setPracticeProbs(probs);
        setIsAnimating(false);
    };

    // Determine which values to show on graph
    // If animation is running (or last interaction was animation), show animation values
    // But if user interacts with practice, switch to practice values
    // Actually, let's simplify: 
    // If user interacts with Practice, we show Practice values.
    // If user clicks Play/Next in Animation, we show Animation values.

    // We can use a timestamp or just a flag. 
    // Let's assume AnimationPanel calls onStepChange when it updates.
    // PracticePanel calls onStepChange when it updates.

    const currentLogits = isAnimating ? animationLogits : practiceLogits;
    const currentProbs = isAnimating ? animationProbs : practiceProbs;

    return (
        <div className="min-h-screen bg-gray-100 p-4">
            <h1 className="text-3xl font-bold text-gray-800 text-center mb-4">Softmax Activation Function</h1>

            <div className="flex flex-col gap-4 max-w-7xl mx-auto">
                {/* Top Row - Animation and Practice */}
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Left Panel - Animation Demo */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden" onClick={() => setIsAnimating(true)}>
                        <SoftmaxAnimationPanel onStepChange={handleAnimationStepChange} />
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
                        <SoftmaxGraphPanel
                            logits={currentLogits}
                            probabilities={currentProbs}
                            isActive={true}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
