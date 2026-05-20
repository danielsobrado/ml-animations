import React, { useState } from 'react';
import DescentPanel from './DescentPanel';
import LandscapePanel from './LandscapePanel';
import VariationsPanel from './VariationsPanel';
import { Tabs } from '../../_design-system/ui';

const TABS = [
    { id: 'descent', label: 'Gradient Descent' },
    { id: 'landscape', label: 'Loss Landscape' },
    { id: 'variations', label: 'Optimizer Variations' }
];

export default function OptimizationAnimation() {
    const [activeTab, setActiveTab] = useState('descent');

    const renderContent = () => {
        switch (activeTab) {
            case 'descent': return <DescentPanel />;
            case 'landscape': return <LandscapePanel />;
            case 'variations': return <VariationsPanel />;
            default: return null;
        }
    };

    return (
        <div className="text-slate-100">
            <div className="max-w-7xl mx-auto">
                <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
