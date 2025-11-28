import React, { useState } from 'react';
import { BookOpen, Hash, BarChart3, Calculator, GraduationCap, Code } from 'lucide-react';
import IntroPanel from './IntroPanel';
import BowPanel from './BowPanel';
import TfIdfPanel from './TfIdfPanel';
import ComparisonPanel from './ComparisonPanel';
import CodePanel from './CodePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'intro', label: '1. Introduction', icon: BookOpen, color: 'blue' },
  { id: 'bow', label: '2. Bag of Words', icon: Hash, color: 'green' },
  { id: 'tfidf', label: '3. TF-IDF', icon: BarChart3, color: 'yellow' },
  { id: 'comparison', label: '4. Comparison', icon: Calculator, color: 'purple' },
  { id: 'code', label: '5. Python Code', icon: Code, color: 'cyan' },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap, color: 'pink' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('intro');

  const renderPanel = () => {
    switch (activeTab) {
      case 'intro': return <IntroPanel />;
      case 'bow': return <BowPanel />;
      case 'tfidf': return <TfIdfPanel />;
      case 'comparison': return <ComparisonPanel />;
      case 'code': return <CodePanel />;
      case 'practice': return <PracticePanel />;
      default: return <IntroPanel />;
    }
  };

  const currentIndex = tabs.findIndex(t => t.id === activeTab);

  return (
    <div className="min-h-screen text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="text-3xl">📝</span>
            <div>
              <h1 className="text-xl font-bold text-white">Bag of Words & TF-IDF</h1>
              <p className="text-xs text-gray-400">Text Representation Fundamentals</p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-black/20 border-b border-white/10 overflow-x-auto">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 py-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                    activeTab === tab.id
                      ? `bg-${tab.color}-600 text-white`
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon size={16} />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {renderPanel()}
      </main>

      {/* Footer Navigation */}
      <footer className="fixed bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm border-t border-white/10 py-3 px-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button
            onClick={() => currentIndex > 0 && setActiveTab(tabs[currentIndex - 1].id)}
            disabled={currentIndex === 0}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ← Previous
          </button>
          <span className="text-gray-400">
            Section <span className="text-white">{currentIndex + 1}</span> of {tabs.length}
          </span>
          <button
            onClick={() => currentIndex < tabs.length - 1 && setActiveTab(tabs[currentIndex + 1].id)}
            disabled={currentIndex === tabs.length - 1}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next →
          </button>
        </div>
      </footer>
    </div>
  );
}
