import React, { useState } from 'react';
import MaxPoolingPanel from './MaxPoolingPanel';
import ConfigPanel from './ConfigPanel';

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

export default function App() {
    const [inputMatrix, setInputMatrix] = useState(generateRandomMatrix());
    const [poolSize, setPoolSize] = useState(2);
    const [stride, setStride] = useState(2);

    const handleParamsChange = (matrix, newPoolSize, newStride) => {
        setInputMatrix(matrix);
        setPoolSize(newPoolSize);
        setStride(newStride);
    };

    return (
        <div className="text-slate-900 dark:text-white">
            <div className="flex flex-col gap-4 max-w-7xl mx-auto">
                {/* Top Row - Animation and Config */}
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Left Panel - Animation */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                        <MaxPoolingPanel
                            inputMatrix={inputMatrix}
                            poolSize={poolSize}
                            stride={stride}
                        />
                    </div>

                    {/* Right Panel - Config */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                        <ConfigPanel onParamsChange={handleParamsChange} />
                    </div>
                </div>
            </div>
        </div>
    );
}
