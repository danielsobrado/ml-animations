import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function HikerPanel() {
    const [position, setPosition] = useState(10); // X position (0-100)
    const [isChecking, setIsChecking] = useState(false);
    const [message, setMessage] = useState("I'm lost in the fog. Help me find the bottom!");

    // Simple valley function: y = (x-50)^2 / 50 + 10
    const getHeight = (x) => Math.pow(x - 50, 2) / 50 + 10;
    const getSlope = (x) => (2 * (x - 50)) / 50; // Derivative

    const checkSlope = () => {
        setIsChecking(true);
        const slope = getSlope(position);

        setTimeout(() => {
            setIsChecking(false);
            if (Math.abs(slope) < 0.1) {
                setMessage("It feels flat here! I think I found the bottom! 🎉");
            } else if (slope > 0) {
                setMessage("Slope is tilting UP to the RIGHT. I should go LEFT.");
            } else {
                setMessage("Slope is tilting UP to the LEFT. I should go RIGHT.");
            }
        }, 1000);
    };

    const takeStep = () => {
        const slope = getSlope(position);
        const stepSize = slope * 5; // Learning rate of 5
        const newPos = Math.max(0, Math.min(100, position - stepSize));
        setPosition(newPos);
        setMessage("Taking a step downhill...");
    };

    return (
        <div className="p-8 h-full flex flex-col items-center">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-indigo-900 mb-4">The Hiker in the Fog</h2>
                <p className="text-lg text-slate-700 leading-relaxed">
                    Imagine you are on a mountain in thick fog. You can't see the bottom.
                    You can only feel the slope with your feet.
                    <strong> Gradient Descent</strong> is just feeling the slope and stepping downhill.
                </p>
            </div>

            {/* Visualization Area */}
            <div className="w-full max-w-4xl h-[400px] bg-sky-100 rounded-2xl border-b-8 border-slate-700 relative overflow-hidden mb-8">
                {/* Fog Overlay */}
                <div className="absolute inset-0 bg-white/60 pointer-events-none z-20 backdrop-blur-[2px]"></div>

                {/* Mountain SVG */}
                <svg className="absolute bottom-0 left-0 w-full h-full z-10" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <path d="M0,100 L0,60 Q50,10 100,60 L100,100 Z" fill="#475569" />
                </svg>

                {/* Hiker */}
                <motion.div
                    className="absolute z-30 flex flex-col items-center"
                    animate={{
                        left: `${position}%`,
                        bottom: `${100 - getHeight(position)}%` // Approximate visual height mapping
                    }}
                    transition={{ type: "spring", stiffness: 50 }}
                    style={{ transform: 'translateX(-50%)' }}
                >
                    {/* Speech Bubble */}
                    <div className="bg-white p-3 rounded-xl shadow-lg mb-2 text-sm font-bold whitespace-nowrap max-w-[200px] text-center">
                        {message}
                    </div>

                    {/* Hiker Emoji */}
                    <div className={`text-6xl transition-transform ${isChecking ? 'rotate-12' : ''}`}>
                        🧗
                    </div>
                </motion.div>
            </div>

            {/* Controls */}
            <div className="flex gap-4">
                <button
                    onClick={checkSlope}
                    disabled={isChecking}
                    className="px-8 py-4 bg-indigo-600 text-white rounded-xl font-bold text-xl hover:bg-indigo-700 shadow-lg disabled:opacity-50 transition-all transform hover:scale-105"
                >
                    🦶 Feel Slope
                </button>
                <button
                    onClick={takeStep}
                    disabled={isChecking}
                    className="px-8 py-4 bg-green-600 text-white rounded-xl font-bold text-xl hover:bg-green-700 shadow-lg disabled:opacity-50 transition-all transform hover:scale-105"
                >
                    👣 Take Step
                </button>
                <button
                    onClick={() => { setPosition(10); setMessage("I'm lost again."); }}
                    className="px-8 py-4 bg-slate-200 text-slate-700 rounded-xl font-bold text-xl hover:bg-slate-300 shadow-lg transition-all"
                >
                    🔄 Reset
                </button>
            </div>
        </div>
    );
}
