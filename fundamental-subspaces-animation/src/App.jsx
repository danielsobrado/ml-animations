import React from 'react';
import AnimationPanel from './AnimationPanel';
import PracticePanel from './PracticePanel';
import RankNullityPanel from './RankNullityPanel';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-100 p-4">
      <h1 className="mb-4 text-center text-3xl font-bold text-slate-900">The Four Fundamental Subspaces</h1>

      <div className="mx-auto flex max-w-7xl flex-col gap-4">
        <div className="rounded-xl bg-slate-50 shadow-lg">
          <AnimationPanel />
        </div>

        <div className="rounded-xl bg-slate-50 shadow-lg">
          <RankNullityPanel />
        </div>

        <div className="rounded-xl bg-slate-50 shadow-lg">
          <PracticePanel />
        </div>
      </div>
    </div>
  );
}
