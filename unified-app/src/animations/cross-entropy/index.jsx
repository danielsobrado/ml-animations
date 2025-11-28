import React, { useState, Suspense, lazy } from 'react';
import { Target, Scale } from 'lucide-react';

// Lazy load panels
const LossPanel = lazy(() => import('./LossPanel'));
const DivergencePanel = lazy(() => import('./DivergencePanel'));

// Tab configuration
const tabs = [
    { id: 'loss', label: '1. Cross-Entropy (The Cost)', icon: Target, color: 'from-orange-500 to-amber-500' },
    { id: 'divergence', label: '2. KL Divergence (The Distance)', icon: Scale, color: 'from-amber-500 to-yellow-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500"></div>
        </div>
    );
}

export default function CrossEntropyAnimation() {
    const [activeTab, setActiveTab] = useState('loss');

    const renderPanel = () => {
        switch (activeTab) {
            case 'loss':
                return <Suspense fallback={<LoadingPanel />}><LossPanel /></Suspense>;
            case 'divergence':
                return <Suspense fallback={<LoadingPanel />}><DivergencePanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><LossPanel /></Suspense>;
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Navigation Tabs */}
            <nav className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10">
                <div className="px-4 overflow-x-auto">
                    <div className="flex space-x-1 py-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                                    activeTab === tab.id
                                        ? `bg-gradient-to-r ${tab.color} text-white shadow-lg scale-105`
                                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-700/50'
                                }`}
                            >
                                <tab.icon size={18} />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </nav>

            {/* Panel Content */}
            <div className="flex-1 overflow-auto">
                {renderPanel()}
            </div>
        </div>
    );
}
