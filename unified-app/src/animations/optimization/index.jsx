import React, { useState } from 'react';
import DescentPanel from './DescentPanel';
import LandscapePanel from './LandscapePanel';
import VariationsPanel from './VariationsPanel';
import { TrendingDown, Activity, Zap } from 'lucide-react';

const TABS = [
    { id: 'descent', label: '1. Gradient Descent', icon: TrendingDown },
    { id: 'landscape', label: '2. Loss Landscape', icon: Activity },
    { id: 'variations', label: '3. Optimizer Variations', icon: Zap }
];

export default function OptimizationAnimation() {
    const [activeTab, setActiveTab] = useState('descent');

    const renderContent = () => {
        switch (activeTab) {
            case 'descent': return <DescentPanel />;
            case 'landscape': return <LandscapePanel />;
            case 'variations': return <VariationsPanel />;
            default: return null;
        }
    };

    return (
        <div className="text-slate-100">
            <div className="max-w-7xl mx-auto">
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {TABS.map(tab => {
                        const Icon = tab.icon;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 ${activeTab === tab.id
                                    ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg scale-105'
                                    : 'bg-slate-800/50 text-slate-700 dark:text-slate-300 hover:bg-slate-700/50 shadow-sm border border-slate-700'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
