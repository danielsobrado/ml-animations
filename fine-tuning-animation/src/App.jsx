import React, { useState } from 'react';
import { Layers, Zap, Database, GraduationCap } from 'lucide-react';
import ConceptPanel from './ConceptPanel';
import LoRAPanel from './LoRAPanel';
import QLoRAPanel from './QLoRAPanel';
import PracticePanel from './PracticePanel';

const tabs = [
    { id: 'concept', label: 'Concept', icon: GraduationCap },
    { id: 'lora', label: 'LoRA', icon: Layers },
    { id: 'qlora', label: 'QLoRA', icon: Zap },
    { id: 'practice', label: 'Practice Lab', icon: Database },
];

export default function App() {
    const [activeTab, setActiveTab] = useState('concept');

    const renderPanel = () => {
        switch (activeTab) {
            case 'concept':
                return <ConceptPanel />;
            case 'lora':
                return <LoRAPanel />;
            case 'qlora':
                return <QLoRAPanel />;
            case 'practice':
                return <PracticePanel />;
            default:
                return <ConceptPanel />;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-50 to-indigo-100">
            {/* Header */}
            <header className="bg-white shadow-sm border-b">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <h1 className="text-2xl font-bold text-purple-900">
                        🎯 Efficient Fine-Tuning
                    </h1>
                    <p className="text-purple-600 text-sm">
                        LoRA, QLoRA & Parameter-Efficient Fine-Tuning (PEFT)
                    </p>
                </div>
            </header>

            {/* Navigation Tabs */}
            <nav className="bg-white border-b">
                <div className="max-w-7xl mx-auto px-4">
                    <div className="flex space-x-1">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                                    activeTab === tab.id
                                        ? 'border-purple-500 text-purple-600 bg-purple-50'
                                        : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'
                                }`}
                            >
                                <tab.icon size={18} />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto">
                {renderPanel()}
            </main>
        </div>
    );
}
