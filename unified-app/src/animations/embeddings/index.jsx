import React, { useState, Suspense, lazy } from 'react';
import { Calculator, Compass, Orbit } from 'lucide-react';

// Lazy load panels
const AlgebraPanel = lazy(() => import('./AlgebraPanel'));
const SimilarityPanel = lazy(() => import('./SimilarityPanel'));
const SpacePanel = lazy(() => import('./SpacePanel'));

// Tab configuration
const tabs = [
    { id: 'algebra', label: '1. Word Algebra', icon: Calculator, color: 'from-cyan-500 to-blue-500' },
    { id: 'similarity', label: '2. Similarity Lab', icon: Compass, color: 'from-purple-500 to-pink-500' },
    { id: 'space', label: '3. 3D Semantic Space', icon: Orbit, color: 'from-indigo-500 to-violet-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
        </div>
    );
}

export default function EmbeddingsAnimation() {
    const [activeTab, setActiveTab] = useState('algebra');

    const renderPanel = () => {
        switch (activeTab) {
            case 'algebra':
                return <Suspense fallback={<LoadingPanel />}><AlgebraPanel /></Suspense>;
            case 'similarity':
                return <Suspense fallback={<LoadingPanel />}><SimilarityPanel /></Suspense>;
            case 'space':
                return <Suspense fallback={<LoadingPanel />}><SpacePanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><AlgebraPanel /></Suspense>;
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
