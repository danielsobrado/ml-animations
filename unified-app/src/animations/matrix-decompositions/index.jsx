import React, { Suspense, lazy, useState } from 'react';
import { FileText, FlaskConical } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const OneSheetPanel = lazy(() => import('./OneSheetPanel'));
const PracticePanel = lazy(() => import('./PracticePanel'));

const tabs = [
  { id: 'sheet', label: '1. One-Sheet', icon: FileText, color: 'from-cyan-500 to-blue-500' },
  { id: 'practice', label: '2. Practice Lab', icon: FlaskConical, color: 'from-rose-500 to-red-500' },
];

function LoadingPanel() {
  return (
    <div className="flex items-center justify-center p-12">
      <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-cyan-500" />
    </div>
  );
}

export default function MatrixDecompositionsAnimation() {
  const [activeTab, setActiveTab] = useState('sheet');

  const renderPanel = () => {
    switch (activeTab) {
      case 'practice':
        return (
          <Suspense fallback={<LoadingPanel />}>
            <PracticePanel />
          </Suspense>
        );
      case 'sheet':
      default:
        return (
          <Suspense fallback={<LoadingPanel />}>
            <OneSheetPanel />
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

      <div className="flex-1 overflow-auto">
        {renderPanel()}
        <div className="mx-auto max-w-7xl px-4 pb-6">
          <AssessmentPanel lessonId="matrix-decompositions" />
        </div>
      </div>
    </div>
  );
}
