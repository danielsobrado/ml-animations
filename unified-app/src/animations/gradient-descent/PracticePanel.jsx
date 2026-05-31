import React, { useState } from 'react';
import { DEFAULT_LEARNING_RATE, DEFAULT_START_WEIGHT, learningRateStatus } from './gradientDescentModel.js';

export default function PracticePanel({
    learningRate: initialLearningRate = DEFAULT_LEARNING_RATE,
    startWeight: initialStartWeight = DEFAULT_START_WEIGHT,
    onParamsChange,
}) {
    const [learningRate, setLearningRate] = useState(initialLearningRate);
    const [startWeight, setStartWeight] = useState(initialStartWeight);

    const handleLRChange = (value) => {
        const nextLearningRate = parseFloat(value);
        setLearningRate(nextLearningRate);
        if (onParamsChange) {
            onParamsChange(nextLearningRate, startWeight);
        }
    };

    const handleWeightChange = (value) => {
        const nextStartWeight = parseFloat(value);
        setStartWeight(nextStartWeight);
        if (onParamsChange) {
            onParamsChange(learningRate, nextStartWeight);
        }
    };

    const status = learningRateStatus(learningRate);

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Controls</h2>

            <div className="flex flex-col gap-6 w-full max-w-md">
                {/* Learning Rate */}
                <div className="flex flex-col gap-2">
                    <label className="font-bold text-gray-700">
                        Learning Rate (alpha):
                    </label>
                    <div className="flex items-center gap-4">
                        <input
                            type="range"
                            min="0.01"
                            max="1.0"
                            step="0.01"
                            value={learningRate}
                            onChange={(e) => handleLRChange(e.target.value)}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <span className="font-mono font-bold w-12 text-right">
                            {learningRate.toFixed(2)}
                        </span>
                    </div>
                    <p className={`text-sm ${status.color} font-bold`}>
                        {status.text}
                    </p>
                </div>

                {/* Starting Weight */}
                <div className="flex flex-col gap-2">
                    <label className="font-bold text-gray-700">
                        Starting Weight:
                    </label>
                    <div className="flex items-center gap-4">
                        <input
                            type="range"
                            min="-5"
                            max="5"
                            step="0.1"
                            value={startWeight}
                            onChange={(e) => handleWeightChange(e.target.value)}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <span className="font-mono font-bold w-12 text-right">
                            {startWeight.toFixed(1)}
                        </span>
                    </div>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 text-sm text-gray-700 space-y-2">
                    <p><strong>Experiment:</strong></p>
                    <ul className="list-disc list-inside space-y-1">
                        <li>Try alpha = 0.01 (slow but steady)</li>
                        <li>Try alpha = 0.5 (fast convergence)</li>
                        <li>Try alpha = 0.95 (watch it oscillate)</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
