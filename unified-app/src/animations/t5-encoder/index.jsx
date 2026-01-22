import React, { useState } from 'react';
import { Brain, Type, Layers, Zap, Scale, FlaskConical } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import TokenizationPanel from './TokenizationPanel';
import ArchitecturePanel from './ArchitecturePanel';
import BidirectionalPanel from './BidirectionalPanel';
import ScalePanel from './ScalePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: 'T5 Overview', icon: Brain },
  { id: 'tokenization', label: 'SentencePiece', icon: Type },
  { id: 'architecture', label: 'Architecture', icon: Layers },
  { id: 'bidirectional', label: 'Bidirectional', icon: Zap },
  { id: 'scale', label: 'Model Scale', icon: Scale },
  { id: 'practice', label: 'Practice Lab', icon: FlaskConical },
];

export default function T5TextEncoderAnimation() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPanel />;
      case 'tokenization':
        return <TokenizationPanel />;
      case 'architecture':
        return <ArchitecturePanel />;
      case 'bidirectional':
        return <BidirectionalPanel />;
      case 'scale':
        return <ScalePanel />;
      case 'practice':
        return <PracticePanel />;
      default:
        return <OverviewPanel />;
    }
  };

  return (
    <div className="text-white">
      <div className="max-w-7xl mx-auto">
        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${activeTab === tab.id
                    ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-500/25'
                    : 'bg-white/10 text-gray-700 dark:text-gray-300 hover:bg-white/20'
                  }`}
              >
                <span className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center text-xs font-bold">
                  {index + 1}
                </span>
                <Icon size={18} />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Panel Content */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 p-6">
          {renderPanel()}
        </div>
      </div>
    </div>
  );
}
