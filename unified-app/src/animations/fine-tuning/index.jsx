import React, { useState, Suspense, lazy } from 'react';
import { GraduationCap, Layers, Zap, Database } from 'lucide-react';

// Lazy load panels
const ConceptPanel = lazy(() => import('./ConceptPanel'));
const LoRAPanel = lazy(() => import('./LoRAPanel'));
const QLoRAPanel = lazy(() => import('./QLoRAPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

// Tab configuration
const tabs = [
    { id: 'concept', label: '1. Concept', icon: GraduationCap, color: 'from-purple-500 to-indigo-500' },
    { id: 'lora', label: '2. LoRA', icon: Layers, color: 'from-blue-500 to-cyan-500' },
    { id: 'qlora', label: '3. QLoRA', icon: Zap, color: 'from-amber-500 to-orange-500' },
    { id: 'practice', label: '4. Practice Lab', icon: Database, color: 'from-rose-500 to-red-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500"></div>
        </div>
    );
}

export default function FineTuningAnimation() {
    const [activeTab, setActiveTab] = useState('concept');

    const renderPanel = () => {
        switch (activeTab) {
            case 'concept':
                return <Suspense fallback={<LoadingPanel />}><ConceptPanel /></Suspense>;
            case 'lora':
                return <Suspense fallback={<LoadingPanel />}><LoRAPanel /></Suspense>;
            case 'qlora':
                return <Suspense fallback={<LoadingPanel />}><QLoRAPanel /></Suspense>;
            case 'practice':
                return <Suspense fallback={<LoadingPanel />}><PracticePanel /></Suspense>;
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
