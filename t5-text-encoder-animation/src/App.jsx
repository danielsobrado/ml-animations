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

function App() {
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-emerald-900/20 to-gray-900 text-white">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent mb-2">
            T5 Text Encoder
          </h1>
          <p className="text-gray-400 text-lg">
            Understanding T5-XXL's role in Stable Diffusion 3
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-500/25'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
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

export default App;
