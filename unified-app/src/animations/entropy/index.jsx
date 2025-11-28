import React, { useState, Suspense, lazy } from 'react';
import { Lightbulb, BarChart3 } from 'lucide-react';

// Lazy load panels
const SurprisePanel = lazy(() => import('./SurprisePanel'));
const EntropyPanel = lazy(() => import('./EntropyPanel'));

// Tab configuration
const tabs = [
    { id: 'surprise', label: '1. The Bit (Surprise)', icon: Lightbulb, color: 'from-pink-500 to-rose-500' },
    { id: 'entropy', label: '2. Entropy (Uncertainty)', icon: BarChart3, color: 'from-rose-500 to-red-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pink-500"></div>
        </div>
    );
}

export default function EntropyAnimation() {
    const [activeTab, setActiveTab] = useState('surprise');

    const renderPanel = () => {
        switch (activeTab) {
            case 'surprise':
                return <Suspense fallback={<LoadingPanel />}><SurprisePanel /></Suspense>;
            case 'entropy':
                return <Suspense fallback={<LoadingPanel />}><EntropyPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><SurprisePanel /></Suspense>;
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
