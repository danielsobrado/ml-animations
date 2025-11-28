import React, { useState, Suspense, lazy } from 'react';
import { Footprints, Network, Scale, FileText } from 'lucide-react';

// Lazy load panels
const PropertyPanel = lazy(() => import('./PropertyPanel'));
const BuilderPanel = lazy(() => import('./BuilderPanel'));
const StationaryPanel = lazy(() => import('./StationaryPanel'));
const TextPanel = lazy(() => import('./TextPanel'));

// Tab configuration
const tabs = [
    { id: 'property', label: '1. The Markov Property', icon: Footprints, color: 'from-blue-500 to-indigo-500' },
    { id: 'builder', label: '2. Transition Matrix', icon: Network, color: 'from-indigo-500 to-violet-500' },
    { id: 'stationary', label: '3. Stationary Distribution', icon: Scale, color: 'from-violet-500 to-purple-500' },
    { id: 'text', label: '4. Text Generation', icon: FileText, color: 'from-purple-500 to-pink-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );
}

export default function MarkovChainsAnimation() {
    const [activeTab, setActiveTab] = useState('property');

    const renderPanel = () => {
        switch (activeTab) {
            case 'property':
                return <Suspense fallback={<LoadingPanel />}><PropertyPanel /></Suspense>;
            case 'builder':
                return <Suspense fallback={<LoadingPanel />}><BuilderPanel /></Suspense>;
            case 'stationary':
                return <Suspense fallback={<LoadingPanel />}><StationaryPanel /></Suspense>;
            case 'text':
                return <Suspense fallback={<LoadingPanel />}><TextPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><PropertyPanel /></Suspense>;
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
