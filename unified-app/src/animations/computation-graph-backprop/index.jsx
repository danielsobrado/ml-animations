import React, { useMemo, useState } from 'react';
import { ArrowRight, Calculator, RotateCcw, Sigma, StepForward } from 'lucide-react';

const tabs = [
  { id: 'forward', label: 'Forward pass', icon: ArrowRight },
  { id: 'backward', label: 'Backward pass', icon: Sigma },
  { id: 'update', label: 'Weight update', icon: StepForward },
];

function relu(value) {
  return Math.max(0, value);
}

function computeGraph({ x, w, b, target, lr }) {
  const wx = w * x;
  const z = wx + b;
  const a = relu(z);
  const error = a - target;
  const loss = 0.5 * error * error;
  const dLossDa = error;
  const dAdZ = z > 0 ? 1 : 0;
  const dZdWx = 1;
  const dWxDw = x;
  const dWxDx = w;
  const dLossDz = dLossDa * dAdZ;
  const dLossDw = dLossDz * dZdWx * dWxDw;
  const dLossDb = dLossDz;
  const dLossDx = dLossDz * dZdWx * dWxDx;
  const nextW = w - lr * dLossDw;
  const nextB = b - lr * dLossDb;
  const nextZ = nextW * x + nextB;
  const nextA = relu(nextZ);
  const nextLoss = 0.5 * (nextA - target) ** 2;

  return {
    wx,
    z,
    a,
    error,
    loss,
    dLossDa,
    dAdZ,
    dZdWx,
    dWxDw,
    dWxDx,
    dLossDz,
    dLossDw,
    dLossDb,
    dLossDx,
    nextW,
    nextB,
    nextLoss,
  };
}

function format(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function Control({ label, min, max, step, value, onChange }) {
  return (
    <label className="block rounded-lg border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between gap-4 text-sm font-semibold text-slate-800">
        <span>{label}</span>
        <span className="font-mono">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-emerald-700"
      />
    </label>
  );
}

function Node({ title, value, gradient, tone = 'slate', active }) {
  const toneClass = {
    blue: 'border-blue-300 bg-blue-50',
    green: 'border-emerald-300 bg-emerald-50',
    orange: 'border-orange-300 bg-orange-50',
    slate: 'border-slate-200 bg-white',
  }[tone];

  return (
    <div className={`rounded-lg border p-4 shadow-sm ${toneClass} ${active ? 'ring-2 ring-slate-900' : ''}`}>
      <div className="text-xs font-bold uppercase tracking-wide text-slate-500">{title}</div>
      <div className="mt-2 text-2xl font-bold text-slate-950">{value}</div>
      {gradient !== undefined && (
        <div className="mt-2 rounded border border-slate-200 bg-white/70 px-2 py-1 font-mono text-xs text-slate-700">
          grad {gradient}
        </div>
      )}
    </div>
  );
}

function Edge({ label, gradient }) {
  return (
    <div className="flex min-h-16 flex-col items-center justify-center gap-1 text-center text-slate-600">
      <ArrowRight size={20} />
      <span className="font-mono text-xs">{label}</span>
      {gradient && <span className="rounded bg-slate-900 px-2 py-1 font-mono text-xs text-white">{gradient}</span>}
    </div>
  );
}

function MobileEdge({ label, gradient }) {
  return (
    <div className="flex items-center justify-center gap-2 text-center text-slate-600">
      <ArrowRight className="rotate-90" size={18} />
      <span className="font-mono text-xs">{label}</span>
      {gradient && <span className="rounded bg-slate-900 px-2 py-1 font-mono text-xs text-white">{gradient}</span>}
    </div>
  );
}

function GraphView({ values, activeTab }) {
  const backward = activeTab !== 'forward';

  return (
    <div className="grid max-w-full min-w-0 gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="grid gap-3 sm:hidden">
        <Node title="input x" value={format(values.x)} gradient={backward ? format(values.dLossDx) : undefined} tone="blue" />
        <MobileEdge label="*" gradient={backward ? `dw path x=${format(values.dWxDw)}` : undefined} />
        <Node title="multiply w*x" value={format(values.wx)} gradient={backward ? format(values.dLossDz) : undefined} />
        <MobileEdge label="+ b" gradient={backward ? `db=${format(values.dLossDb)}` : undefined} />
        <Node title="pre-activation z" value={format(values.z)} gradient={backward ? format(values.dLossDz) : undefined} tone="orange" />
        <MobileEdge label="ReLU" gradient={backward ? `local=${format(values.dAdZ)}` : undefined} />
        <Node title="activation a" value={format(values.a)} gradient={backward ? format(values.dLossDa) : undefined} tone="green" />
        <MobileEdge label="compare" gradient={backward ? `a-y=${format(values.error)}` : undefined} />
        <Node title="loss" value={format(values.loss)} tone="orange" active={activeTab === 'update'} />
      </div>

      <div className="hidden overflow-x-auto sm:grid">
        <div className="grid min-w-[760px] grid-cols-[1fr_80px_1fr_80px_1fr_80px_1fr] items-center">
          <Node title="input x" value={format(values.x)} gradient={backward ? format(values.dLossDx) : undefined} tone="blue" />
          <Edge label="*" gradient={backward ? `dw path x=${format(values.dWxDw)}` : undefined} />
          <Node title="multiply w*x" value={format(values.wx)} gradient={backward ? format(values.dLossDz) : undefined} />
          <Edge label="+ b" gradient={backward ? `db=${format(values.dLossDb)}` : undefined} />
          <Node title="pre-activation z" value={format(values.z)} gradient={backward ? format(values.dLossDz) : undefined} tone="orange" />
          <Edge label="ReLU" gradient={backward ? `local=${format(values.dAdZ)}` : undefined} />
          <Node title="activation a" value={format(values.a)} gradient={backward ? format(values.dLossDa) : undefined} tone="green" />
        </div>
        <div className="grid min-w-[760px] grid-cols-[1fr_80px_1fr] items-center">
          <div />
          <Edge label="compare" gradient={backward ? `a-y=${format(values.error)}` : undefined} />
          <Node title="loss" value={format(values.loss)} tone="orange" active={activeTab === 'update'} />
        </div>
      </div>
    </div>
  );
}

function FormulaPanel({ values, activeTab }) {
  if (activeTab === 'forward') {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h3 className="text-lg font-bold text-slate-950">Forward pass</h3>
        <div className="mt-3 space-y-2 font-mono text-sm text-slate-700">
          <p>w*x = {format(values.w)} * {format(values.x)} = {format(values.wx)}</p>
          <p>z = w*x + b = {format(values.wx)} + {format(values.b)} = {format(values.z)}</p>
          <p>a = ReLU(z) = {format(values.a)}</p>
          <p>L = 0.5 * (a - y)^2 = {format(values.loss)}</p>
        </div>
      </div>
    );
  }

  if (activeTab === 'backward') {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4">
        <h3 className="text-lg font-bold text-slate-950">Reverse accumulation</h3>
        <div className="mt-3 space-y-2 font-mono text-sm text-slate-700">
          <p>dL/da = a - y = {format(values.dLossDa)}</p>
          <p>da/dz = ReLU'(z) = {format(values.dAdZ)}</p>
          <p>dL/dw = dL/da * da/dz * dz/dw = {format(values.dLossDw)}</p>
          <p>dL/db = dL/da * da/dz * dz/db = {format(values.dLossDb)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-lg font-bold text-slate-950">One optimizer step</h3>
      <div className="mt-3 space-y-2 font-mono text-sm text-slate-700">
        <p>w' = w - lr * dL/dw = {format(values.nextW)}</p>
        <p>b' = b - lr * dL/db = {format(values.nextB)}</p>
        <p>loss before = {format(values.loss)}</p>
        <p>loss after = {format(values.nextLoss)}</p>
      </div>
    </div>
  );
}

function InsightPanel({ values, activeTab }) {
  const reluBlocked = values.dAdZ === 0;
  const lossFalls = values.nextLoss < values.loss;

  return (
    <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-emerald-950">
      <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-emerald-800">
        <Calculator size={16} />
        Check yourself
      </div>
      {activeTab === 'forward' && (
        <p className="mt-3 leading-7">
          Predict the loss before changing any slider. The graph only stores simple local computations, but together they define the whole training objective.
        </p>
      )}
      {activeTab === 'backward' && (
        <p className="mt-3 leading-7">
          {reluBlocked
            ? 'Because z is below zero, ReLU blocks the gradient. This is the tiny version of why activation choices matter.'
            : 'Each backward number is a local derivative multiplied by the upstream gradient. That multiplication is the chain rule doing the work.'}
        </p>
      )}
      {activeTab === 'update' && (
        <p className="mt-3 leading-7">
          {lossFalls
            ? 'The update moves parameters in the negative-gradient direction, so this step lowers loss for the current example.'
            : 'This step does not lower loss. Try a smaller learning rate or move z into the active ReLU region.'}
        </p>
      )}
    </div>
  );
}

export default function ComputationGraphBackpropAnimation() {
  const [activeTab, setActiveTab] = useState('forward');
  const [params, setParams] = useState({
    x: 1.4,
    w: 0.8,
    b: -0.2,
    target: 1.6,
    lr: 0.25,
  });
  const values = useMemo(() => computeGraph(params), [params]);
  const mergedValues = { ...params, ...values };

  const reset = () => {
    setParams({ x: 1.4, w: 0.8, b: -0.2, target: 1.6, lr: 0.25 });
    setActiveTab('forward');
  };

  const setParam = (key) => (value) => setParams((current) => ({ ...current, [key]: value }));

  return (
    <div className="min-h-full bg-[#fbf8f1] text-slate-900">
      <nav className="sticky top-0 z-10 border-b border-slate-200 bg-[#fbf8f1]/95 backdrop-blur">
        <div className="grid grid-cols-2 gap-2 px-4 py-3 sm:flex sm:overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`flex min-h-11 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition sm:px-4 ${
                activeTab === tab.id ? 'bg-slate-900 text-white' : 'bg-white text-slate-700 hover:bg-slate-100'
              }`}
            >
              <tab.icon size={17} />
              {tab.label}
            </button>
          ))}
          <button type="button" onClick={reset} className="flex min-h-11 items-center justify-center gap-2 rounded-lg bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 sm:ml-auto sm:px-4">
            <RotateCcw size={17} />
            Reset
          </button>
        </div>
      </nav>

      <section className="mx-auto grid max-w-6xl min-w-0 gap-5 px-4 py-8">
        <div className="min-w-0">
          <div className="text-xs font-bold uppercase tracking-wide text-emerald-800">Training loop bridge</div>
          <h2 className="mt-2 text-2xl font-bold text-slate-950 sm:text-3xl">Computation Graph & Backpropagation</h2>
          <p className="mt-3 max-w-3xl text-base leading-8 text-slate-700 sm:text-lg">
            A forward pass computes predictions and loss. Backpropagation walks the same graph backward,
            multiplying local derivatives so each parameter receives the gradient it needs for an update.
          </p>
        </div>

        <div className="grid min-w-0 gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
          <aside className="grid min-w-0 gap-3 self-start">
            <Control label="Input x" min={-2} max={2} step={0.1} value={params.x} onChange={setParam('x')} />
            <Control label="Weight w" min={-2} max={2} step={0.1} value={params.w} onChange={setParam('w')} />
            <Control label="Bias b" min={-2} max={2} step={0.1} value={params.b} onChange={setParam('b')} />
            <Control label="Target y" min={-1} max={3} step={0.1} value={params.target} onChange={setParam('target')} />
            <Control label="Learning rate" min={0.05} max={0.8} step={0.05} value={params.lr} onChange={setParam('lr')} />
          </aside>

          <main className="grid min-w-0 gap-5">
            <GraphView values={mergedValues} activeTab={activeTab} />
            <div className="grid min-w-0 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)]">
              <FormulaPanel values={mergedValues} activeTab={activeTab} />
              <InsightPanel values={mergedValues} activeTab={activeTab} />
            </div>
          </main>
        </div>
      </section>
    </div>
  );
}
