import React, { useState, Suspense, lazy } from 'react';
import { BookOpen, Microscope, PlayCircle, Repeat } from 'lucide-react';

// Lazy load panels
const ConceptPanel = lazy(() => import('./ConceptPanel'));
const AnatomyPanel = lazy(() => import('./AnatomyPanel'));
const FlowPanel = lazy(() => import('./FlowPanel'));
const SequencePanel = lazy(() => import('./SequencePanel'));

// Tab configuration
const tabs = [
    { id: 'concept', label: '1. The Conveyor Belt', icon: BookOpen, color: 'from-indigo-500 to-violet-500' },
    { id: 'anatomy', label: '2. Anatomy Lab', icon: Microscope, color: 'from-blue-500 to-cyan-500' },
    { id: 'flow', label: '3. Bit-by-Bit Flow', icon: PlayCircle, color: 'from-green-500 to-emerald-500' },
    { id: 'sequence', label: '4. Sequence View', icon: Repeat, color: 'from-purple-500 to-pink-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        </div>
    );
}

export default function LstmAnimation() {
    const [activeTab, setActiveTab] = useState('concept');

    const renderPanel = () => {
        switch (activeTab) {
            case 'concept':
                return <Suspense fallback={<LoadingPanel />}><ConceptPanel /></Suspense>;
            case 'anatomy':
                return <Suspense fallback={<LoadingPanel />}><AnatomyPanel /></Suspense>;
            case 'flow':
                return <Suspense fallback={<LoadingPanel />}><FlowPanel /></Suspense>;
            case 'sequence':
                return <Suspense fallback={<LoadingPanel />}><SequencePanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><ConceptPanel /></Suspense>;
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
