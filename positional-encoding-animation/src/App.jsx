import React, { useState } from 'react';
import ProblemPanel from './ProblemPanel';
import SinusoidalPanel from './SinusoidalPanel';
import PlaygroundPanel from './PlaygroundPanel';
import { AlertCircle, Waves, Sparkles } from 'lucide-react';

const TABS = [
    { id: 'problem', label: '1. The Problem', icon: AlertCircle },
    { id: 'sinusoidal', label: '2. Sinusoidal Encoding', icon: Waves },
    { id: 'playground', label: '3. Encoding Playground', icon: Sparkles }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('problem');

    const renderContent = () => {
        switch (activeTab) {
            case 'problem': return <ProblemPanel />;
            case 'sinusoidal': return <SinusoidalPanel />;
            case 'playground': return <PlaygroundPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400 mb-2 tracking-tight">
                        Positional Encoding
                    </h1>
                    <p className="text-slate-300 text-lg">
                        How Transformers understand word order.
                    </p>
                </header>

                {/* Navigation Tabs */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {TABS.map(tab => {
                        const Icon = tab.icon;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 ${activeTab === tab.id
                                        ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg scale-105'
                                        : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50 shadow-sm border border-slate-700'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
