import React, { useState } from 'react';
import { Activity, TrendingDown, Zap } from 'lucide-react';
import { Tabs } from './_design-system/ui';
import DescentPanel from './DescentPanel';
import LandscapePanel from './LandscapePanel';
import VariationsPanel from './VariationsPanel';

const tabs = [
  { id: 'descent', label: '1. Descent', icon: TrendingDown },
  { id: 'landscape', label: '2. Landscape', icon: Activity },
  { id: 'variations', label: '3. Variations', icon: Zap },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('descent');

  const renderContent = () => {
    switch (activeTab) {
      case 'landscape':
        return <LandscapePanel />;
      case 'variations':
        return <VariationsPanel />;
      case 'descent':
      default:
        return <DescentPanel />;
    }
  };

  return (
    <div className="ds-page">
      <div className="ds-shell">
        <header className="ds-header">
          <div className="ds-eyebrow">
            <span>Standalone</span>
            <span className="sep">/</span>
            <span>Optimization</span>
          </div>
          <h1 className="ds-title">Optimization</h1>
          <p className="ds-subtitle">How models learn by moving across a loss landscape.</p>
        </header>

        <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

        <section className="ds-panel">
          <div className="ds-panel-body">{renderContent()}</div>
        </section>
      </div>
    </div>
  );
}
