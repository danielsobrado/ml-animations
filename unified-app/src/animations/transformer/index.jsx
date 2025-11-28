import React, { useState, Suspense, lazy } from 'react';
import { Building2, Layers, ArrowRightLeft, Zap, GraduationCap } from 'lucide-react';

// Lazy load panels
const OverviewPanel = lazy(() => import('./OverviewPanel'));
const EncoderPanel = lazy(() => import('./EncoderPanel'));
const DecoderPanel = lazy(() => import('./DecoderPanel'));
const DataFlowPanel = lazy(() => import('./DataFlowPanel'));
const VariantsPanel = lazy(() => import('./VariantsPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

// Tab configuration
const tabs = [
    { id: 'overview', label: '1. Architecture', icon: Building2, color: 'from-amber-500 to-orange-500' },
    { id: 'encoder', label: '2. Encoder', icon: Layers, color: 'from-blue-500 to-cyan-500' },
    { id: 'decoder', label: '3. Decoder', icon: Layers, color: 'from-purple-500 to-pink-500' },
    { id: 'dataflow', label: '4. Data Flow', icon: ArrowRightLeft, color: 'from-green-500 to-emerald-500' },
    { id: 'variants', label: '5. Variants', icon: Zap, color: 'from-indigo-500 to-violet-500' },
    { id: 'practice', label: '6. Practice Lab', icon: GraduationCap, color: 'from-rose-500 to-red-500' },
];

// Loading fallback
function LoadingPanel() {
    return (
        <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-500"></div>
        </div>
    );
}

export default function TransformerAnimation() {
    const [activeTab, setActiveTab] = useState('overview');

    const renderPanel = () => {
        switch (activeTab) {
            case 'overview':
                return <Suspense fallback={<LoadingPanel />}><OverviewPanel /></Suspense>;
            case 'encoder':
                return <Suspense fallback={<LoadingPanel />}><EncoderPanel /></Suspense>;
            case 'decoder':
                return <Suspense fallback={<LoadingPanel />}><DecoderPanel /></Suspense>;
            case 'dataflow':
                return <Suspense fallback={<LoadingPanel />}><DataFlowPanel /></Suspense>;
            case 'variants':
                return <Suspense fallback={<LoadingPanel />}><VariantsPanel /></Suspense>;
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
