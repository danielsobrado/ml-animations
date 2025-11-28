import React, { useState, Suspense, lazy } from 'react';
import { Play, AlertTriangle, Settings } from 'lucide-react';

// Lazy load panels
const PlaygroundPanel = lazy(() => import('./PlaygroundPanel'));
const CollisionPanel = lazy(() => import('./CollisionPanel'));
const TuningPanel = lazy(() => import('./TuningPanel'));

// Tab configuration
const tabs = [
    { id: 'playground', label: '1. Playground', icon: Play, color: 'from-indigo-500 to-violet-500' },
    { id: 'collision', label: '2. False Positive Lab', icon: AlertTriangle, color: 'from-amber-500 to-orange-500' },
    { id: 'tuning', label: '3. Tuning Studio', icon: Settings, color: 'from-blue-500 to-cyan-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        </div>
    );
}

export default function BloomFilterAnimation() {
    const [activeTab, setActiveTab] = useState('playground');

    const renderPanel = () => {
        switch (activeTab) {
            case 'playground':
                return <Suspense fallback={<LoadingPanel />}><PlaygroundPanel /></Suspense>;
            case 'collision':
                return <Suspense fallback={<LoadingPanel />}><CollisionPanel /></Suspense>;
            case 'tuning':
                return <Suspense fallback={<LoadingPanel />}><TuningPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><PlaygroundPanel /></Suspense>;
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
