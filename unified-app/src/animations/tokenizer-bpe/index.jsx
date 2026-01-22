import React, { useState } from 'react';
import { Type, Hash, Layers, GitCompare, Workflow, FlaskConical } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import BPEPanel from './BPEPanel';
import VocabularyPanel from './VocabularyPanel';
import ComparisonPanel from './ComparisonPanel';
import PipelinePanel from './PipelinePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: 'Overview', icon: Type },
  { id: 'bpe', label: 'BPE Algorithm', icon: Hash },
  { id: 'vocabulary', label: 'Vocabulary', icon: Layers },
  { id: 'comparison', label: 'CLIP vs T5', icon: GitCompare },
  { id: 'pipeline', label: 'Full Pipeline', icon: Workflow },
  { id: 'practice', label: 'Practice Lab', icon: FlaskConical },
];

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPanel />;
      case 'bpe':
        return <BPEPanel />;
      case 'vocabulary':
        return <VocabularyPanel />;
      case 'comparison':
        return <ComparisonPanel />;
      case 'pipeline':
        return <PipelinePanel />;
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
                    ? 'bg-gradient-to-r from-orange-600 to-red-600 text-white shadow-lg shadow-orange-500/25'
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

export default App;
