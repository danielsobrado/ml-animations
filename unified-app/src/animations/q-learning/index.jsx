import React, { useState, Suspense, lazy } from 'react';
import { Grid, Calculator, PlayCircle } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

// Lazy load panels
const TablePanel = lazy(() => import('./TablePanel'));
const AlgorithmPanel = lazy(() => import('./AlgorithmPanel'));
const TrainingPanel = lazy(() => import('./TrainingPanel'));

// Tab configuration
const tabs = [
    { id: 'table', label: '1. The Q-Table (Brain)', icon: Grid, color: 'from-cyan-500 to-blue-500' },
    { id: 'algorithm', label: '2. The Bellman Update', icon: Calculator, color: 'from-blue-500 to-indigo-500' },
    { id: 'training', label: '3. Training Loop', icon: PlayCircle, color: 'from-indigo-500 to-violet-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
        </div>
    );
}

export default function QLearningAnimation() {
    const [activeTab, setActiveTab] = useState('table');

    const renderPanel = () => {
        switch (activeTab) {
            case 'table':
                return <Suspense fallback={<LoadingPanel />}><TablePanel /></Suspense>;
            case 'algorithm':
                return <Suspense fallback={<LoadingPanel />}><AlgorithmPanel /></Suspense>;
            case 'training':
                return <Suspense fallback={<LoadingPanel />}><TrainingPanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><TablePanel /></Suspense>;
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Navigation Tabs */}
            <nav className="bg-white/50 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-10">
                <div className="px-4 overflow-x-auto">
                    <div className="flex space-x-1 py-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                type="button"
                                aria-selected={activeTab === tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                                    activeTab === tab.id
                                        ? 'is-active'
                                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
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
                <div className="px-8 pb-8">
                    <AssessmentPanel lessonId="q-learning" title="Q-Learning check" />
                </div>
            </div>
        </div>
    );
}
