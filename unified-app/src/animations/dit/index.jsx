import React, { useState } from 'react';
import { Cpu, Layers, Wand2, Settings, Scale, GraduationCap } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import BlockStructurePanel from './BlockStructurePanel';
import AdaLNPanel from './AdaLNPanel';
import ConditioningPanel from './ConditioningPanel';
import ScalingPanel from './ScalingPanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: '1. DiT Overview', icon: Cpu },
  { id: 'block', label: '2. Block Structure', icon: Layers },
  { id: 'adaln', label: '3. AdaLN-Zero', icon: Wand2 },
  { id: 'conditioning', label: '4. Conditioning', icon: Settings },
  { id: 'scaling', label: '5. Scaling Laws', icon: Scale },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPanel />;
      case 'block': return <BlockStructurePanel />;
      case 'adaln': return <AdaLNPanel />;
      case 'conditioning': return <ConditioningPanel />;
      case 'scaling': return <ScalingPanel />;
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
                  ? 'bg-gradient-to-r from-pink-600 to-orange-600 text-white'
                  : isCompleted
                    ? 'bg-pink-500/30 text-pink-300'
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
