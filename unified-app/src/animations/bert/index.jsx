import React, { useState, Suspense, lazy } from 'react';
import { BookOpen, Type, Layers, Target, Cpu, Brain, Settings, Zap, GraduationCap } from 'lucide-react';

// Lazy load panels
const OverviewPanel = lazy(() => import('./OverviewPanel'));
const TokenizationPanel = lazy(() => import('./TokenizationPanel'));
const EmbeddingsPanel = lazy(() => import('./EmbeddingsPanel'));
const AttentionPanel = lazy(() => import('./AttentionPanel'));
const EncoderLayerPanel = lazy(() => import('./EncoderLayerPanel'));
const PreTrainingPanel = lazy(() => import('./PreTrainingPanel'));
const FineTuningPanel = lazy(() => import('./FineTuningPanel'));
const TasksPanel = lazy(() => import('./TasksPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

// Tab configuration
const tabs = [
    { id: 'overview', label: '1. Overview', icon: BookOpen, color: 'from-blue-500 to-cyan-500' },
    { id: 'tokenization', label: '2. Tokenization', icon: Type, color: 'from-green-500 to-emerald-500' },
    { id: 'embeddings', label: '3. Embeddings', icon: Layers, color: 'from-amber-500 to-yellow-500' },
    { id: 'attention', label: '4. Self-Attention', icon: Target, color: 'from-purple-500 to-pink-500' },
    { id: 'encoder', label: '5. Encoder Layers', icon: Cpu, color: 'from-pink-500 to-rose-500' },
    { id: 'pretraining', label: '6. Pre-training', icon: Brain, color: 'from-orange-500 to-red-500' },
    { id: 'finetuning', label: '7. Fine-tuning', icon: Settings, color: 'from-cyan-500 to-teal-500' },
    { id: 'examples', label: '8. Live Examples', icon: Zap, color: 'from-red-500 to-rose-500' },
    { id: 'practice', label: '9. Practice Lab', icon: GraduationCap, color: 'from-indigo-500 to-violet-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );
}

export default function BertAnimation() {
    const [activeTab, setActiveTab] = useState('overview');

    const renderPanel = () => {
        switch (activeTab) {
            case 'overview':
                return <Suspense fallback={<LoadingPanel />}><OverviewPanel /></Suspense>;
            case 'tokenization':
                return <Suspense fallback={<LoadingPanel />}><TokenizationPanel /></Suspense>;
            case 'embeddings':
                return <Suspense fallback={<LoadingPanel />}><EmbeddingsPanel /></Suspense>;
            case 'attention':
                return <Suspense fallback={<LoadingPanel />}><AttentionPanel /></Suspense>;
            case 'encoder':
                return <Suspense fallback={<LoadingPanel />}><EncoderLayerPanel /></Suspense>;
            case 'pretraining':
                return <Suspense fallback={<LoadingPanel />}><PreTrainingPanel /></Suspense>;
            case 'finetuning':
                return <Suspense fallback={<LoadingPanel />}><FineTuningPanel /></Suspense>;
            case 'examples':
                return <Suspense fallback={<LoadingPanel />}><TasksPanel /></Suspense>;
            case 'practice':
                return <Suspense fallback={<LoadingPanel />}><PracticePanel /></Suspense>;
            default:
                return <Suspense fallback={<LoadingPanel />}><OverviewPanel /></Suspense>;
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
