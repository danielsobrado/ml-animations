import React, { useState, Suspense, lazy } from 'react';
import { Gamepad2, Coins, TrendingUp } from 'lucide-react';

// Lazy load panels
const AgentPanel = lazy(() => import('./AgentPanel'));
const RewardPanel = lazy(() => import('./RewardPanel'));
const ReturnPanel = lazy(() => import('./ReturnPanel'));

// Tab configuration
const tabs = [
    { id: 'agent', label: '1. The Agent & Environment', icon: Gamepad2, color: 'from-emerald-500 to-teal-500' },
    { id: 'reward', label: '2. Rewards & Penalties', icon: Coins, color: 'from-teal-500 to-cyan-500' },
    { id: 'return', label: '3. Discounted Returns', icon: TrendingUp, color: 'from-cyan-500 to-blue-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
        </div>
    );
}

export default function RlFoundationsAnimation() {
    const [activeTab, setActiveTab] = useState('agent');

    const renderPanel = () => {
        switch (activeTab) {
            case 'agent':
                return <Suspense fallback={<LoadingPanel />}><AgentPanel /></Suspense>;
            case 'reward':
                return <Suspense fallback={<LoadingPanel />}><RewardPanel /></Suspense>;
            case 'return':
                return <Suspense fallback={<LoadingPanel />}><ReturnPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><AgentPanel /></Suspense>;
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
