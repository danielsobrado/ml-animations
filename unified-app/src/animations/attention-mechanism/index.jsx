import React, { useState, Suspense, lazy } from 'react';
import { Lightbulb, Brain, Calculator, Grid3X3, CheckCircle, Eye, Sparkles } from 'lucide-react';

// Lazy load panels
const IntuitionPanel = lazy(() => import('./IntuitionPanel'));

// Tab configuration
const tabs = [
    { id: 'intuition', label: '1. Intuition', icon: Lightbulb, color: 'from-amber-500 to-orange-500' },
    { id: 'qkv', label: '2. Q, K, V', icon: Brain, color: 'from-blue-500 to-cyan-500' },
    { id: 'scaled', label: '3. Scaled Dot-Product', icon: Calculator, color: 'from-purple-500 to-pink-500' },
    { id: 'multihead', label: '4. Multi-Head', icon: Grid3X3, color: 'from-green-500 to-emerald-500' },
    { id: 'self', label: '5. Self-Attention', icon: Eye, color: 'from-indigo-500 to-violet-500' },
    { id: 'practice', label: '6. Practice Lab', icon: CheckCircle, color: 'from-rose-500 to-red-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );
}

// Placeholder for panels not yet integrated
function PlaceholderPanel({ tabId, tabLabel }) {
    return (
        <div className="p-6 text-center">
            <div className="card p-8 max-w-lg mx-auto">
                <div className="text-4xl mb-4">🚧</div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                    {tabLabel} Panel
                </h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">
                    This panel is being integrated from the original animation.
                    Check back soon!
                </p>
            </div>
        </div>
    );
}

export default function AttentionMechanismAnimation() {
    const [activeTab, setActiveTab] = useState('intuition');

    const renderPanel = () => {
        switch (activeTab) {
            case 'intuition':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <IntuitionPanel />
                    </Suspense>
                );
            case 'qkv':
            case 'scaled':
            case 'multihead':
            case 'self':
            case 'practice':
                const tab = tabs.find(t => t.id === activeTab);
                return <PlaceholderPanel tabId={activeTab} tabLabel={tab?.label || activeTab} />;
            default:
                return <IntuitionPanel />;
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

            {/* Progress Indicator */}
            <div className="bg-slate-100/50 dark:bg-slate-800/30 border-b border-slate-200 dark:border-slate-700">
                <div className="px-4 py-2">
                    <div className="flex items-center gap-2">
                        {tabs.map((tab, i) => (
                            <React.Fragment key={tab.id}>
                                <div 
                                    className={`w-3 h-3 rounded-full transition-all cursor-pointer ${
                                        activeTab === tab.id 
                                            ? `bg-gradient-to-r ${tab.color} animate-pulse` 
                                            : tabs.findIndex(t => t.id === activeTab) > i
                                                ? 'bg-green-500'
                                                : 'bg-slate-300 dark:bg-slate-600'
                                    }`}
                                    onClick={() => setActiveTab(tab.id)}
                                    title={tab.label}
                                />
                                {i < tabs.length - 1 && (
                                    <div className={`flex-1 h-0.5 ${
                                        tabs.findIndex(t => t.id === activeTab) > i
                                            ? 'bg-green-500'
                                            : 'bg-slate-300 dark:bg-slate-600'
                                    }`} />
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            </div>

            {/* Panel Content */}
            <div className="flex-1 overflow-y-auto">
                {renderPanel()}
            </div>
        </div>
    );
}
