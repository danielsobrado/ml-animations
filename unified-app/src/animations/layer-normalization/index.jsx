import React, { useState, Suspense, lazy } from 'react';
import { TrendingDown, Scale, GitBranch } from 'lucide-react';

// Lazy load panels
const ProblemPanel = lazy(() => import('./ProblemPanel'));
const ComparisonPanel = lazy(() => import('./ComparisonPanel'));
const ResidualPanel = lazy(() => import('./ResidualPanel'));

// Tab configuration
const tabs = [
    { id: 'problem', label: '1. The Problem', icon: TrendingDown, color: 'from-violet-500 to-purple-500' },
    { id: 'comparison', label: '2. Layer vs Batch Norm', icon: Scale, color: 'from-fuchsia-500 to-pink-500' },
    { id: 'residual', label: '3. Residual Connections', icon: GitBranch, color: 'from-pink-500 to-rose-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-500"></div>
        </div>
    );
}

export default function LayerNormalizationAnimation() {
    const [activeTab, setActiveTab] = useState('problem');

    const renderPanel = () => {
        switch (activeTab) {
            case 'problem':
                return <Suspense fallback={<LoadingPanel />}><ProblemPanel /></Suspense>;
            case 'comparison':
                return <Suspense fallback={<LoadingPanel />}><ComparisonPanel /></Suspense>;
            case 'residual':
                return <Suspense fallback={<LoadingPanel />}><ResidualPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><ProblemPanel /></Suspense>;
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
