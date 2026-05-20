import React, { Suspense, lazy, useState } from 'react';
import { FlaskConical, Play, SplitSquareHorizontal } from 'lucide-react';

const AnimationPanel = lazy(() => import('./AnimationPanel'));
const RankNullityPanel = lazy(() => import('./RankNullityPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

const tabs = [
  { id: 'animation', label: '1. Animation', icon: Play, color: 'from-indigo-500 to-cyan-500' },
  { id: 'rank-nullity', label: '2. Rank-Nullity', icon: SplitSquareHorizontal, color: 'from-emerald-500 to-teal-500' },
  { id: 'practice', label: '3. Practice Lab', icon: FlaskConical, color: 'from-rose-500 to-red-500' },
];

function LoadingPanel() {
  return (
    <div className="flex items-center justify-center p-12">
      <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-indigo-500" />
    </div>
  );
}

export default function FundamentalSubspacesAnimation() {
  const [activeTab, setActiveTab] = useState('animation');

  const renderPanel = () => {
    switch (activeTab) {
      case 'rank-nullity':
        return (
          <Suspense fallback={<LoadingPanel />}>
            <RankNullityPanel />
          </Suspense>
        );
      case 'practice':
        return (
          <Suspense fallback={<LoadingPanel />}>
            <PracticePanel />
          </Suspense>
        );
      case 'animation':
      default:
        return (
          <Suspense fallback={<LoadingPanel />}>
            <AnimationPanel />
          </Suspense>
        );
    }
  };

  return (
    <div className="flex h-full flex-col">
      <nav className="sticky top-0 z-10 border-b border-slate-200 bg-white/50 backdrop-blur-sm">
        <div className="overflow-x-auto px-4">
          <div className="flex space-x-1 py-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 whitespace-nowrap rounded-xl px-4 py-2.5 text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? `bg-gradient-to-r ${tab.color} scale-105 text-white shadow-lg`
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <tab.icon size={18} />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <div className="flex-1 overflow-auto">{renderPanel()}</div>
    </div>
  );
}
