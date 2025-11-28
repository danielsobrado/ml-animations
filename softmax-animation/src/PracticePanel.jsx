import React, { useState, useEffect } from 'react';

export default function PracticePanel({ onStepChange }) {
    const [logits, setLogits] = useState([2.0, 1.0, 0.1]);

    const handleChange = (index, value) => {
        const newLogits = [...logits];
        newLogits[index] = parseFloat(value);
        setLogits(newLogits);
    };

    useEffect(() => {
        const exponentials = logits.map(z => Math.exp(z));
        const sumExp = exponentials.reduce((a, b) => a + b, 0);
        const probabilities = exponentials.map(e => e / sumExp);

        if (onStepChange) {
            onStepChange(logits, probabilities);
        }
    }, [logits, onStepChange]);

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Interactive Practice</h2>

            <div className="flex flex-col gap-4 w-full max-w-md">
                {logits.map((val, i) => (
                    <div key={i} className="flex items-center gap-4">
                        <label className="font-bold text-gray-700 w-20">Logit z{i + 1}:</label>
                        <input
                            type="range"
                            min="-5"
                            max="5"
                            step="0.1"
                            value={val}
                            onChange={(e) => handleChange(i, e.target.value)}
                            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <input
                            type="number"
                            value={val}
                            onChange={(e) => handleChange(i, e.target.value)}
                            className="w-16 p-1 border rounded text-center"
                            step="0.1"
                        />
                    </div>
                ))}
            </div>

            <div className="mt-6 p-4 bg-blue-50 rounded-lg text-sm text-blue-800">
                <p>Adjust the logits to see how probabilities change.</p>
                <p className="mt-1">Notice that increasing one logit decreases the probabilities of others!</p>
            </div>
        </div>
    );
}
