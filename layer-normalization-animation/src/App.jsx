import React, { useState } from 'react';
import ProblemPanel from './ProblemPanel';
import ComparisonPanel from './ComparisonPanel';
import ResidualPanel from './ResidualPanel';
import { TrendingDown, Scale, GitBranch } from 'lucide-react';

const TABS = [
    { id: 'problem', label: '1. The Problem', icon: TrendingDown },
    { id: 'comparison', label: '2. Layer vs Batch Norm', icon: Scale },
    { id: 'residual', label: '3. Residual Connections', icon: GitBranch }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('problem');

    const renderContent = () => {
        switch (activeTab) {
            case 'problem': return <ProblemPanel />;
            case 'comparison': return <ComparisonPanel />;
            case 'residual': return <ResidualPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-violet-950 via-purple-900 to-fuchsia-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-fuchsia-400 to-pink-400 mb-2 tracking-tight">
                        Layer Normalization
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Stabilizing deep network training.
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
                                        ? 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-lg scale-105'
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
