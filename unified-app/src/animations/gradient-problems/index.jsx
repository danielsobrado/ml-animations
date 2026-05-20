import React, { useState } from 'react';
import ChainPanel from './ChainPanel';
import SolutionPanel from './SolutionPanel';
import { Tabs } from '../../_design-system/ui';

const TABS = [
    { id: 'chain', label: 'The Chain of Destruction' },
    { id: 'solution', label: 'The Residual Fix' }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('chain');

    const renderContent = () => {
        switch (activeTab) {
            case 'chain': return <ChainPanel />;
            case 'solution': return <SolutionPanel />;
            default: return null;
        }
    };

    return (
        <div className="text-slate-100">
            <div className="max-w-7xl mx-auto">
                <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

                {/* Main Content Area */}
                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
