import React, { useState } from 'react';

export default function ConfigPanel({ onParamsChange }) {
    const [poolSize, setPoolSize] = useState(2);
    const [stride, setStride] = useState(2);

    const generateRandomMatrix = () => {
        const size = 4;
        const matrix = [];
        for (let i = 0; i < size; i++) {
            const row = [];
            for (let j = 0; j < size; j++) {
                row.push(Math.floor(Math.random() * 10));
            }
            matrix.push(row);
        }
        return matrix;
    };

    const [inputMatrix, setInputMatrix] = useState(generateRandomMatrix());

    const handlePoolSizeChange = (value) => {
        setPoolSize(parseInt(value));
        if (onParamsChange) {
            onParamsChange(inputMatrix, parseInt(value), stride);
        }
    };

    const handleStrideChange = (value) => {
        setStride(parseInt(value));
        if (onParamsChange) {
            onParamsChange(inputMatrix, poolSize, parseInt(value));
        }
    };

    const handleRandomize = () => {
        const newMatrix = generateRandomMatrix();
        setInputMatrix(newMatrix);
        if (onParamsChange) {
            onParamsChange(newMatrix, poolSize, stride);
        }
    };

    const inputSize = inputMatrix.length;
    const outputSize = Math.floor((inputSize - poolSize) / stride) + 1;

    return (
        <div className="flex flex-col items-center p-4 h-full">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Configuration</h2>

            <div className="flex flex-col gap-6 w-full max-w-md">
                {/* Pool Size */}
                <div className="flex flex-col gap-2">
                    <label className="font-bold text-gray-700">
                        Pool Size:
                    </label>
                    <div className="flex gap-2">
                        {[2, 3].map((size) => (
                            <button
                                key={size}
                                onClick={() => handlePoolSizeChange(size)}
                                className={`px-4 py-2 rounded-lg font-bold transition-colors ${poolSize === size
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                    }`}
                            >
                                {size}×{size}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Stride */}
                <div className="flex flex-col gap-2">
                    <label className="font-bold text-gray-700">
                        Stride:
                    </label>
                    <div className="flex gap-2">
                        {[1, 2].map((s) => (
                            <button
                                key={s}
                                onClick={() => handleStrideChange(s)}
                                className={`px-4 py-2 rounded-lg font-bold transition-colors ${stride === s
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                    }`}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Dimensions */}
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                    <p className="text-sm font-bold text-gray-700 mb-2">Dimensions:</p>
                    <p className="text-sm text-gray-800 dark:text-gray-600">
                        Input: <span className="font-mono font-bold">{inputSize}×{inputSize}</span>
                    </p>
                    <p className="text-sm text-gray-800 dark:text-gray-600">
                        Output: <span className="font-mono font-bold">{outputSize}×{outputSize}</span>
                    </p>
                    <p className="text-xs text-gray-700 dark:text-gray-500 mt-2">
                        Formula: (Input - Pool) / Stride + 1
                    </p>
                </div>

                {/* Randomize Button */}
                <button
                    onClick={handleRandomize}
                    className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white font-bold rounded-lg transition-colors"
                >
                    🎲 Randomize Input
                </button>

                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200 text-sm text-gray-700 space-y-2">
                    <p><strong>How Max Pooling Works:</strong></p>
                    <ul className="list-disc list-inside space-y-1">
                        <li>Slide window across input</li>
                        <li>Take maximum value in each window</li>
                        <li>Downsamples feature maps (reduces size)</li>
                        <li>Preserves strongest features</li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
