import React, { useState, Suspense, lazy } from 'react';
import { Share2, Waves, Calculator } from 'lucide-react';

// Lazy load panels
const GraphCanvas = lazy(() => import('./GraphCanvas'));
const SurferPanel = lazy(() => import('./SurferPanel'));
const IterativePanel = lazy(() => import('./IterativePanel'));

// Tab configuration
const tabs = [
    { id: 'builder', label: '1. Graph Builder', icon: Share2, color: 'from-indigo-500 to-violet-500' },
    { id: 'surfer', label: '2. Random Surfer', icon: Waves, color: 'from-blue-500 to-cyan-500' },
    { id: 'iterative', label: '3. Power Method', icon: Calculator, color: 'from-green-500 to-emerald-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        </div>
    );
}

export default function PagerankAnimation() {
    const [activeTab, setActiveTab] = useState('builder');
    
    // Shared graph state
    const [nodes, setNodes] = useState([
        { id: 'A', x: 200, y: 200 },
        { id: 'B', x: 400, y: 200 },
        { id: 'C', x: 300, y: 400 }
    ]);
    const [links, setLinks] = useState([
        { source: 'A', target: 'B' },
        { source: 'B', target: 'C' },
        { source: 'C', target: 'A' }
    ]);

    const renderPanel = () => {
        switch (activeTab) {
            case 'builder':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <GraphCanvas nodes={nodes} setNodes={setNodes} links={links} setLinks={setLinks} />
                    </Suspense>
                );
            case 'surfer':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <SurferPanel nodes={nodes} links={links} />
                    </Suspense>
                );
            case 'iterative':
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <IterativePanel nodes={nodes} links={links} />
                    </Suspense>
                );
            default:
                return (
                    <Suspense fallback={<LoadingPanel />}>
                        <GraphCanvas nodes={nodes} setNodes={setNodes} links={links} setLinks={setLinks} />
                    </Suspense>
                );
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
