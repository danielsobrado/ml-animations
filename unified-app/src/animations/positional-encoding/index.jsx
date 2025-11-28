import React, { useState, Suspense, lazy } from 'react';
import { AlertCircle, Waves, Sparkles } from 'lucide-react';

// Lazy load panels
const ProblemPanel = lazy(() => import('./ProblemPanel'));
const SinusoidalPanel = lazy(() => import('./SinusoidalPanel'));
const PlaygroundPanel = lazy(() => import('./PlaygroundPanel'));

// Tab configuration
const tabs = [
    { id: 'problem', label: '1. The Problem', icon: AlertCircle, color: 'from-blue-500 to-cyan-500' },
    { id: 'sinusoidal', label: '2. Sinusoidal Encoding', icon: Waves, color: 'from-cyan-500 to-teal-500' },
    { id: 'playground', label: '3. Encoding Playground', icon: Sparkles, color: 'from-teal-500 to-green-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );
}

export default function PositionalEncodingAnimation() {
    const [activeTab, setActiveTab] = useState('problem');

    const renderPanel = () => {
        switch (activeTab) {
            case 'problem':
                return <Suspense fallback={<LoadingPanel />}><ProblemPanel /></Suspense>;
            case 'sinusoidal':
                return <Suspense fallback={<LoadingPanel />}><SinusoidalPanel /></Suspense>;
            case 'playground':
                return <Suspense fallback={<LoadingPanel />}><PlaygroundPanel /></Suspense>;
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
