import React, { useState } from 'react';
import PlaygroundPanel from './PlaygroundPanel';
import CollisionPanel from './CollisionPanel';
import TuningPanel from './TuningPanel';
import { Play, AlertTriangle, Settings } from 'lucide-react';

const TABS = [
    { id: 'playground', label: '1. Playground', icon: Play },
    { id: 'collision', label: '2. False Positive Lab', icon: AlertTriangle },
    { id: 'tuning', label: '3. Tuning Studio', icon: Settings }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('playground');

    const renderContent = () => {
        switch (activeTab) {
            case 'playground': return <PlaygroundPanel />;
            case 'collision': return <CollisionPanel />;
            case 'tuning': return <TuningPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">
                        Bloom Filter
                    </h1>
                    <p className="text-slate-600 text-lg">
                        Probabilistic Data Structures: Space vs. Accuracy
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
                                        ? 'bg-indigo-600 text-white shadow-lg scale-105'
                                        : 'bg-white text-slate-600 hover:bg-slate-100 shadow-sm border border-slate-200'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
