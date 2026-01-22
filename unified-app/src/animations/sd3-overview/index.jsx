import React, { useState } from 'react';
import { Cpu, FileText, Image, Layers, Workflow, GraduationCap } from 'lucide-react';
import ArchitecturePanel from './ArchitecturePanel';
import TextEncodingPanel from './TextEncodingPanel';
import LatentSpacePanel from './LatentSpacePanel';
import DiffusionProcessPanel from './DiffusionProcessPanel';
import InferencePanel from './InferencePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'architecture', label: '1. Full Architecture', icon: Cpu },
  { id: 'text', label: '2. Text Encoding', icon: FileText },
  { id: 'latent', label: '3. Latent Space', icon: Layers },
  { id: 'diffusion', label: '4. Diffusion Process', icon: Workflow },
  { id: 'inference', label: '5. Inference Pipeline', icon: Image },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('architecture');

  const renderPanel = () => {
    switch (activeTab) {
      case 'architecture': return <ArchitecturePanel />;
      case 'text': return <TextEncodingPanel />;
      case 'latent': return <LatentSpacePanel />;
      case 'diffusion': return <DiffusionProcessPanel />;
      case 'inference': return <InferencePanel />;
      case 'practice': return <PracticePanel />;
      default: return <ArchitecturePanel />;
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
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${isActive
                  ? 'bg-fuchsia-600 text-white'
                  : isCompleted
                    ? 'bg-fuchsia-900/50 text-fuchsia-300 hover:bg-fuchsia-800/50'
                    : 'bg-white/5 text-gray-800 dark:text-gray-400 hover:bg-white/10 hover:text-white'
                }`}
            >
              <Icon size={18} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Progress Bar */}
      <div className="h-1 bg-black/20 flex">
        {tabs.map((tab, index) => {
          const tabIndex = tabs.findIndex(t => t.id === activeTab);
          return (
            <div
              key={tab.id}
              className={`flex-1 transition-all duration-300 ${index <= tabIndex
                  ? 'bg-gradient-to-r from-fuchsia-500 to-purple-600'
                  : 'bg-transparent'
                }`}
            />
          );
        })}
      </div>

      {/* Main Content */}
      <main className="p-6 max-w-7xl mx-auto">
        {renderPanel()}
      </main>

    </div>
  );
}
