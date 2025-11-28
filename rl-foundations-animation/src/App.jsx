import React, { useState } from 'react';
import AgentPanel from './AgentPanel';
import RewardPanel from './RewardPanel';
import ReturnPanel from './ReturnPanel';
import { Gamepad2, Coins, TrendingUp } from 'lucide-react';

const TABS = [
    { id: 'agent', label: '1. The Agent & Environment', icon: Gamepad2 },
    { id: 'reward', label: '2. Rewards & Penalties', icon: Coins },
    { id: 'return', label: '3. Discounted Returns', icon: TrendingUp }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('agent');

    const renderContent = () => {
        switch (activeTab) {
            case 'agent': return <AgentPanel />;
            case 'reward': return <RewardPanel />;
            case 'return': return <ReturnPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-emerald-950 to-teal-950 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 mb-2 tracking-tight">
                        RL Foundations
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Part 1: Agents, Environments, and Rewards.
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
                                        ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg scale-105'
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
