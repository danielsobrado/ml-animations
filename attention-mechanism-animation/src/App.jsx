import React, { useState } from 'react';
import { Lightbulb, Brain, Calculator, Grid3X3, CheckCircle, Eye, Sparkles } from 'lucide-react';
import IntuitionPanel from './IntuitionPanel';
import QKVPanel from './QKVPanel';
import ScaledDotProductPanel from './ScaledDotProductPanel';
import MultiHeadPanel from './MultiHeadPanel';
import SelfAttentionPanel from './SelfAttentionPanel';
import PracticePanel from './PracticePanel';

const tabs = [
    { id: 'intuition', label: '1. Intuition', icon: Lightbulb, color: 'from-amber-500 to-orange-500' },
    { id: 'qkv', label: '2. Q, K, V', icon: Brain, color: 'from-blue-500 to-cyan-500' },
    { id: 'scaled', label: '3. Scaled Dot-Product', icon: Calculator, color: 'from-purple-500 to-pink-500' },
    { id: 'multihead', label: '4. Multi-Head', icon: Grid3X3, color: 'from-green-500 to-emerald-500' },
    { id: 'self', label: '5. Self-Attention', icon: Eye, color: 'from-indigo-500 to-violet-500' },
    { id: 'practice', label: '6. Practice Lab', icon: CheckCircle, color: 'from-rose-500 to-red-500' },
];

export default function App() {
    const [activeTab, setActiveTab] = useState('intuition');

    const renderPanel = () => {
        switch (activeTab) {
            case 'intuition':
                return <IntuitionPanel />;
            case 'qkv':
                return <QKVPanel />;
            case 'scaled':
                return <ScaledDotProductPanel />;
            case 'multihead':
                return <MultiHeadPanel />;
            case 'self':
                return <SelfAttentionPanel />;
            case 'practice':
                return <PracticePanel />;
            default:
                return <IntuitionPanel />;
        }
    };

    return (
        <div className="min-h-screen">
            {/* Header */}
            <header className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center gap-3">
                        <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl">
                            <Sparkles className="text-white" size={28} />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-white">
                                Attention Mechanism
                            </h1>
                            <p className="text-slate-400 text-sm">
                                "Attention Is All You Need" - The foundation of Transformers
                            </p>
                        </div>
                    </div>
                </div>
            </header>

            {/* Navigation Tabs */}
            <nav className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
                <div className="max-w-7xl mx-auto px-4">
                    <div className="flex space-x-1 overflow-x-auto py-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${
                                    activeTab === tab.id
                                        ? `bg-gradient-to-r ${tab.color} text-white shadow-lg scale-105`
                                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
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
            <div className="bg-slate-800/30 border-b border-slate-700">
                <div className="max-w-7xl mx-auto px-4 py-2">
                    <div className="flex items-center gap-2">
                        {tabs.map((tab, i) => (
                            <React.Fragment key={tab.id}>
                                <div 
                                    className={`w-3 h-3 rounded-full transition-all cursor-pointer ${
                                        activeTab === tab.id 
                                            ? `bg-gradient-to-r ${tab.color} animate-pulse` 
                                            : tabs.findIndex(t => t.id === activeTab) > i
                                                ? 'bg-green-500'
                                                : 'bg-slate-600'
                                    }`}
                                    onClick={() => setActiveTab(tab.id)}
                                />
                                {i < tabs.length - 1 && (
                                    <div className={`flex-1 h-0.5 ${
                                        tabs.findIndex(t => t.id === activeTab) > i
                                            ? 'bg-green-500'
                                            : 'bg-slate-600'
                                    }`} />
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto">
                {renderPanel()}
            </main>

            {/* Footer Tip */}
            <footer className="bg-slate-900/50 border-t border-slate-700 mt-8">
                <div className="max-w-7xl mx-auto px-4 py-3 text-center text-slate-500 text-sm">
                    💡 Tip: Progress through each tab in order for the best learning experience
                </div>
            </footer>
        </div>
    );
}
