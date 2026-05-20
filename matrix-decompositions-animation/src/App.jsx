import React from 'react';
import OneSheetPanel from './OneSheetPanel';
import PracticePanel from './PracticePanel';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-100 p-4">
      <h1 className="mb-4 text-center text-3xl font-bold text-slate-900">Matrix Decompositions One-Sheet</h1>

      <div className="mx-auto flex max-w-7xl flex-col gap-4">
        <div className="rounded-xl bg-slate-50 shadow-lg">
          <OneSheetPanel />
        </div>

        <div className="rounded-xl bg-slate-50 shadow-lg">
          <PracticePanel />
        </div>
      </div>
    </div>
  );
}
