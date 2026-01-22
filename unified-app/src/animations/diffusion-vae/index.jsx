import React, { useState } from 'react';
import { Box, Layers, Target, Shuffle, GitBranch, GraduationCap } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import EncoderPanel from './EncoderPanel';
import LatentSpacePanel from './LatentSpacePanel';
import DecoderPanel from './DecoderPanel';
import LossPanel from './LossPanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: '1. Architecture', icon: Box },
  { id: 'encoder', label: '2. Encoder', icon: Layers },
  { id: 'latent', label: '3. Latent Space', icon: Target },
  { id: 'decoder', label: '4. Decoder', icon: Shuffle },
  { id: 'loss', label: '5. Loss Function', icon: GitBranch },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPanel />;
      case 'encoder': return <EncoderPanel />;
      case 'latent': return <LatentSpacePanel />;
      case 'decoder': return <DecoderPanel />;
      case 'loss': return <LossPanel />;
      case 'practice': return <PracticePanel />;
      default: return <OverviewPanel />;
    }
  };

  return (
    <div className="min-h-screen text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-xl">
            🎲
          </div>
          <div>
            <h1 className="text-xl font-bold">Variational Autoencoder (VAE)</h1>
            <p className="text-sm text-gray-800 dark:text-gray-400">Learning to Generate: Probabilistic Latent Representations</p>
          </div>
        </div>
      </header>

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
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                isActive
                  ? 'bg-purple-600 text-white'
                  : isCompleted
                  ? 'bg-purple-900/50 text-purple-300 hover:bg-purple-800/50'
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
              className={`flex-1 transition-all duration-300 ${
                index <= tabIndex
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500'
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

      {/* Footer */}
      <footer className="text-center py-4 text-gray-700 dark:text-sm">
        💡 Tip: Progress through each tab in order for the best learning experience
      </footer>
    </div>
  );
}
