import React, { useState } from 'react';
import ExpectedValuePanel from './ExpectedValuePanel';
import VariancePanel from './VariancePanel';
import DecisionPanel from './DecisionPanel';
import { Scale, Ruler, TrendingUp } from 'lucide-react';

const TABS = [
    { id: 'expected', label: '1. Expected Value', icon: Scale },
    { id: 'variance', label: '2. Variance', icon: Ruler },
    { id: 'decision', label: '3. Decision Making', icon: TrendingUp }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('expected');

    const renderContent = () => {
        switch (activeTab) {
            case 'expected': return <ExpectedValuePanel />;
            case 'variance': return <VariancePanel />;
            case 'decision': return <DecisionPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-amber-950 via-orange-900 to-red-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 mb-2 tracking-tight">
                        Expected Value & Variance
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Quantifying center and spread.
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
                                        ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg scale-105'
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
