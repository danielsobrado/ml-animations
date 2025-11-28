import React, { useState } from 'react';
import DiscretePanel from './DiscretePanel';
import ContinuousPanel from './ContinuousPanel';
import ComparisonPanel from './ComparisonPanel';
import { Dices, Activity, Scale } from 'lucide-react';

const TABS = [
    { id: 'discrete', label: '1. Discrete Distributions', icon: Dices },
    { id: 'continuous', label: '2. Continuous Distributions', icon: Activity },
    { id: 'comparison', label: '3. PMF vs PDF', icon: Scale }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('discrete');

    const renderContent = () => {
        switch (activeTab) {
            case 'discrete': return <DiscretePanel />;
            case 'continuous': return <ContinuousPanel />;
            case 'comparison': return <ComparisonPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-purple-900 to-pink-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 mb-2 tracking-tight">
                        Probability Distributions
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Modeling randomness with mathematics.
                    </p>
                </header>

                {/* Navigation Tabs */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {TABS.map(tab => {
                        const Icon = tab.icon;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 ${activeTab === tab.id
                                        ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg scale-105'
                                        : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50 shadow-sm border border-slate-700'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
