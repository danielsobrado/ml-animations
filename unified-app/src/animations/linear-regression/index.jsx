import React, { useState, Suspense, lazy } from 'react';
import { Ruler, MousePointer2, TrendingUp } from 'lucide-react';

// Lazy load panels
const ResidualsPanel = lazy(() => import('./ResidualsPanel'));
const InteractivePanel = lazy(() => import('./InteractivePanel'));
const CostPanel = lazy(() => import('./CostPanel'));

// Tab configuration
const tabs = [
    { id: 'residuals', label: '1. The Residuals', icon: Ruler, color: 'from-indigo-500 to-violet-500' },
    { id: 'interactive', label: '2. Interactive Fitter', icon: MousePointer2, color: 'from-blue-500 to-cyan-500' },
    { id: 'cost', label: '3. Cost Landscape', icon: TrendingUp, color: 'from-green-500 to-emerald-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        </div>
    );
}

export default function LinearRegressionAnimation() {
    const [activeTab, setActiveTab] = useState('residuals');

    const renderPanel = () => {
        switch (activeTab) {
            case 'residuals':
                return <Suspense fallback={<LoadingPanel />}><ResidualsPanel /></Suspense>;
            case 'interactive':
                return <Suspense fallback={<LoadingPanel />}><InteractivePanel /></Suspense>;
            case 'cost':
                return <Suspense fallback={<LoadingPanel />}><CostPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><ResidualsPanel /></Suspense>;
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
