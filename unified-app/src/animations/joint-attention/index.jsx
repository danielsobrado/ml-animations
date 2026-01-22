import React, { useState } from 'react';
import { Combine, Split, Layers, ArrowLeftRight, Calculator, GraduationCap } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import ConcatPanel from './ConcatPanel';
import AttentionFlowPanel from './AttentionFlowPanel';
import BidirectionalPanel from './BidirectionalPanel';
import ComputationPanel from './ComputationPanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: '1. Why Joint?', icon: Combine },
  { id: 'concat', label: '2. Token Concatenation', icon: Layers },
  { id: 'flow', label: '3. Attention Flow', icon: ArrowLeftRight },
  { id: 'bidirectional', label: '4. Bidirectional Fusion', icon: Split },
  { id: 'computation', label: '5. Computation', icon: Calculator },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPanel />;
      case 'concat': return <ConcatPanel />;
      case 'flow': return <AttentionFlowPanel />;
      case 'bidirectional': return <BidirectionalPanel />;
      case 'computation': return <ComputationPanel />;
      case 'practice': return <PracticePanel />;
      default: return <OverviewPanel />;
    }
  };

  return (
    <div className="text-white">
      {/* Tab Navigation */}
      <nav className="bg-black/20 px-6 py-2 flex gap-2 overflow-x-auto">
        {tabs.map((tab, index) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          const tabIndex = tabs.findIndex(t => t.id === activeTab);
          const isCompleted = index < tabIndex;

          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${isActive
                  ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white'
                  : isCompleted
                    ? 'bg-violet-500/30 text-violet-300'
                    : 'bg-white/5 text-gray-800 dark:text-gray-400 hover:bg-white/10'
                }`}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Content */}
      <main className="p-6 max-w-7xl mx-auto">
        {renderPanel()}
      </main>
    </div>
  );
}
