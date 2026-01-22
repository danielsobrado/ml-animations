import React, { useState } from 'react';
import { Workflow, Zap, LineChart, Calculator, Clock, GraduationCap } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import FlowConceptPanel from './FlowConceptPanel';
import EulerSchedulerPanel from './EulerSchedulerPanel';
import LogitNormalPanel from './LogitNormalPanel';
import SigmaSchedulePanel from './SigmaSchedulePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: '1. What is Flow Matching?', icon: Workflow },
  { id: 'flow', label: '2. Flow Concept', icon: Zap },
  { id: 'euler', label: '3. Euler Scheduler', icon: LineChart },
  { id: 'logit', label: '4. Logit-Normal Sampling', icon: Calculator },
  { id: 'sigma', label: '5. Sigma Schedules', icon: Clock },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPanel />;
      case 'flow': return <FlowConceptPanel />;
      case 'euler': return <EulerSchedulerPanel />;
      case 'logit': return <LogitNormalPanel />;
      case 'sigma': return <SigmaSchedulePanel />;
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
      <div className="max-w-7xl mx-auto">
        {/* Tab Navigation */}
        <nav className="px-6 py-2 flex gap-2 overflow-x-auto">
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
        <div className="h-1 flex">
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
        <main className="p-6">
          {renderPanel()}
        </main>
      </div>
    </div>
  );
}
