import React, { useState } from 'react';
import ChainPanel from './ChainPanel';
import SolutionPanel from './SolutionPanel';
import { Link, ShieldCheck } from 'lucide-react';

const TABS = [
    { id: 'chain', label: '1. The Chain of Destruction', icon: Link },
    { id: 'solution', label: '2. The Residual Fix', icon: ShieldCheck }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('chain');

    const renderContent = () => {
        switch (activeTab) {
            case 'chain': return <ChainPanel />;
            case 'solution': return <SolutionPanel />;
            default: return null;
        }
    };

    return (
        <div className="text-slate-100">
            <div className="max-w-7xl mx-auto">
                {/* Navigation Tabs */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {TABS.map(tab => {
                        const Icon = tab.icon;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 ${activeTab === tab.id
                                    ? 'bg-gradient-to-r from-red-600 to-orange-600 text-white shadow-lg scale-105'
                                    : 'bg-slate-800/50 text-slate-700 dark:text-slate-300 hover:bg-slate-700/50 shadow-sm border border-slate-700'
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
