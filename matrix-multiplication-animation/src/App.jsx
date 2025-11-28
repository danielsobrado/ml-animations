import React from 'react';
import AnimationPanel from './AnimationPanel';
import PracticePanel from './PracticePanel';

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold text-gray-800 text-center mb-4">Matrix Multiplication</h1>
      
      <div className="flex flex-col lg:flex-row gap-4 max-w-7xl mx-auto">
        {/* Left Panel - Animation Demo */}
        <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
          <AnimationPanel />
        </div>
        
        {/* Right Panel - Interactive Practice */}
        <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
          <PracticePanel />
        </div>
      </div>
    </div>
  );
}
