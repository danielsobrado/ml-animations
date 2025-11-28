import React, { useState, Suspense, lazy } from 'react';
import { Calculator, Film, Search } from 'lucide-react';

// Lazy load panels
const DotProductPanel = lazy(() => import('./DotProductPanel'));
const RecommenderPanel = lazy(() => import('./RecommenderPanel'));
const SearchPanel = lazy(() => import('./SearchPanel'));

// Tab configuration
const tabs = [
    { id: 'dot', label: '1. The Dot Product', icon: Calculator, color: 'from-cyan-500 to-purple-500' },
    { id: 'recommender', label: '2. Movie Matcher', icon: Film, color: 'from-purple-500 to-pink-500' },
    { id: 'search', label: '3. Search Engine', icon: Search, color: 'from-pink-500 to-rose-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
        </div>
    );
}

export default function CosineSimilarityAnimation() {
    const [activeTab, setActiveTab] = useState('dot');

    const renderPanel = () => {
        switch (activeTab) {
            case 'dot':
                return <Suspense fallback={<LoadingPanel />}><DotProductPanel /></Suspense>;
            case 'recommender':
                return <Suspense fallback={<LoadingPanel />}><RecommenderPanel /></Suspense>;
            case 'search':
                return <Suspense fallback={<LoadingPanel />}><SearchPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><DotProductPanel /></Suspense>;
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
