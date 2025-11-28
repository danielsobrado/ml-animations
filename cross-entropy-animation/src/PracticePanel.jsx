import React, { useState, useEffect } from 'react';

export default function PracticePanel({ onStepChange }) {
    const [probability, setProbability] = useState(0.7);

    useEffect(() => {
        if (onStepChange) {
            onStepChange(probability);
        }
    }, [probability, onStepChange]);

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Interactive Practice</h2>

            <div className="flex flex-col gap-6 w-full max-w-md">
                <div className="flex flex-col gap-2">
                    <label className="font-bold text-gray-700">
                        Probability of Correct Class (p):
                    </label>
                    <div className="flex items-center gap-4">
                        <input
                            type="range"
                            min="0.01"
                            max="0.99"
                            step="0.01"
                            value={probability}
                            onChange={(e) => setProbability(parseFloat(e.target.value))}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <span className="font-mono font-bold w-12 text-right">
                            {probability.toFixed(2)}
                        </span>
                    </div>
                </div>

                <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                    <p className="text-center text-lg">
                        Loss = -ln({probability.toFixed(2)}) =
                        <span className="font-bold text-orange-600 ml-2">
                            {(-Math.log(probability)).toFixed(2)}
                        </span>
                    </p>
                </div>

                <div className="text-sm text-gray-600 space-y-2">
                    <p>Try dragging the slider to the left (p → 0).</p>
                    <p>Notice how the Loss explodes? This is how the model is punished for being wrong!</p>
                </div>
            </div>
        </div>
    );
}
