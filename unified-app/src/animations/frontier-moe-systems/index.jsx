import React, { useMemo, useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Network,
  RotateCcw,
  Server,
  Zap,
  Gauge,
  TrendingUp,
  Sparkles,
  RefreshCw,
  Flame
} from 'lucide-react';
import {
  MOE_PRESETS,
  TOKEN_DOMAINS,
  FAILURE_MODES,
  PAPER_SIGNAL_CARDS
} from './data';

const TABS = [
  { id: 'why-moe', label: 'Why Sparse MoE?', icon: Zap },
  { id: 'dense-vs-moe', label: 'Dense vs Sparse', icon: Layers },
  { id: 'active-vs-total', label: 'Parameter Budget', icon: Database },
  { id: 'top-k-routing', label: 'Routing Gating', icon: ShuffleIcon },
  { id: 'shared-vs-routed', label: 'Shared Experts', icon: GitBranch },
  { id: 'load-balancing', label: 'Load Balancing', icon: ScaleIcon },
  { id: 'expert-parallelism', label: 'Expert Parallelism', icon: Server },
  { id: 'failure-modes', label: 'Failure Debugger', icon: AlertTriangle },
  { id: 'expert-specialization', label: 'Expert Specialization', icon: Brain },
  { id: 'distillation', label: 'MoE Distillation', icon: Sparkles }
];

function ShuffleIcon(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22" />
      <path d="m18 2 4 4-4 4" />
      <path d="M2 6h1.9c1.2 0 2.3.6 3 1.7l1.1 1.6" />
      <path d="m15.4 12.8 1.2 1.7c.8 1.1 2 1.7 3.2 1.7H22" />
      <path d="m18 14 4 4-4 4" />
    </svg>
  );
}

function ScaleIcon(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="m16 16 3-8 3 8c-.2.7-1 1.8-3 1.8s-2.8-1.1-3-1.8z" />
      <path d="m2 16 3-8 3 8c-.2.7-1 1.8-3 1.8s-2.8-1.1-3-1.8z" />
      <path d="M7 21h10" />
      <path d="M12 3v18" />
      <path d="M3 7h18" />
    </svg>
  );
}

const STREAM_TOKENS = [
  { text: 'The', domain: 'common', color: 'border-blue-300 bg-blue-50 text-blue-950' },
  { text: 'proof', domain: 'common', color: 'border-blue-300 bg-blue-50 text-blue-950' },
  { text: 'uses', domain: 'common', color: 'border-blue-300 bg-blue-50 text-blue-950' },
  { text: 'Python', domain: 'code', color: 'border-emerald-300 bg-emerald-50 text-emerald-950' },
  { text: 'to', domain: 'common', color: 'border-blue-300 bg-blue-50 text-blue-950' },
  { text: 'solve', domain: 'math', color: 'border-purple-300 bg-purple-50 text-purple-950' },
  { text: 'integral', domain: 'math', color: 'border-purple-300 bg-purple-50 text-purple-950' },
  { text: 'bonjour', domain: 'multilingual', color: 'border-orange-300 bg-orange-50 text-orange-950' },
  { text: 'contract', domain: 'legal', color: 'border-rose-300 bg-rose-50 text-rose-950' },
  { text: 'tensor', domain: 'ml', color: 'border-pink-300 bg-pink-50 text-pink-950' }
];

export default function FrontierMoESystems() {
  const [activeTab, setActiveTab] = useState('why-moe');
  const [selectedPreset, setSelectedPreset] = useState('deepseekV3');

  // Animation state for the Expert Router Workbench
  const [tokenIndex, setTokenIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [workbenchSpeed, setWorkbenchSpeed] = useState(1500); // ms per token

  // Sliders and parameter inputs
  const [routingTemp, setRoutingTemp] = useState(0.7);
  const [capacityFactor, setCapacityFactor] = useState(1.2);
  const [networkLatency, setNetworkLatency] = useState(25); // ms
  const [auxLossWeight, setAuxLossWeight] = useState(0.02);
  const [numExpertsActive, setNumExpertsActive] = useState(8);
  const [numSharedExperts, setNumSharedExperts] = useState(1);
  const [routingNoise, setRoutingNoise] = useState(0.1);

  // Distillation parameters
  const [teacherSoftWeight, setTeacherSoftWeight] = useState(0.6);
  const [routerDistillWeight, setRouterDistillWeight] = useState(0.3);
  const [distillEpochs, setDistillEpochs] = useState(20);
  const [isDistilling, setIsDistilling] = useState(false);
  const [distillProgress, setDistillProgress] = useState(0);
  const [distillMetrics, setDistillMetrics] = useState({ studentLoss: 2.8, routerKl: 0.9 });

  // Failure debugger fixes applied
  const [debuggerFixes, setDebuggerFixes] = useState({
    'expert-collapse': false,
    'dead-experts': false,
    'token-dropping': false,
    'communication-bottleneck': false
  });

  const preset = MOE_PRESETS[selectedPreset];

  // Auto-play the token stream workbench
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setTokenIndex((prev) => (prev + 1) % STREAM_TOKENS.length);
    }, workbenchSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, workbenchSpeed]);

  const activeTokenObj = STREAM_TOKENS[tokenIndex];

  // Simulated live KPIs for the Router Workbench
  const kpiMetrics = useMemo(() => {
    const isCollapse = !debuggerFixes['expert-collapse'] && activeTab === 'failure-modes';
    const isDropping = !debuggerFixes['token-dropping'] && (capacityFactor < 1.0 || activeTab === 'failure-modes');
    const isBottleneck = !debuggerFixes['communication-bottleneck'] && (networkLatency > 40 || activeTab === 'failure-modes');
    
    let flOpts = (preset.activeParamsB / preset.totalParamsB) * 100;
    let networkDispatch = preset.activeRoutedExperts * 8.5 * (networkLatency / 10);
    let straggler = isBottleneck ? 35 + networkLatency * 0.8 : 5 + networkLatency * 0.15;
    let droppingPenalty = isDropping ? 18.4 : 0.0;
    
    if (isCollapse) {
      straggler += 28.5;
    }

    return {
      flopsUtilized: flOpts.toFixed(1) + '%',
      stragglerDelay: straggler.toFixed(1) + 'ms',
      droppingRate: droppingPenalty.toFixed(1) + '%',
      dispatchSize: networkDispatch.toFixed(1) + ' GB/s'
    };
  }, [selectedPreset, networkLatency, capacityFactor, debuggerFixes, activeTab, preset]);

  // Handle distillation run simulation
  const handleRunDistillation = () => {
    setIsDistilling(true);
    setDistillProgress(0);
    
    let step = 0;
    const interval = setInterval(() => {
      step += 5;
      setDistillProgress(step);
      
      setDistillMetrics(prev => ({
        studentLoss: Math.max(1.2, 2.8 - (step / 100) * 1.5 * (1 + teacherSoftWeight * 0.2)).toFixed(2),
        routerKl: Math.max(0.08, 0.9 - (step / 100) * 0.8 * (1 + routerDistillWeight * 0.5)).toFixed(3)
      }));

      if (step >= 100) {
        clearInterval(interval);
        setIsDistilling(false);
      }
    }, 150);
  };

  const MathBlock = ({ children }) => (
    <span className="font-mono px-1.5 py-0.5 rounded border text-xs bg-slate-100 border-slate-200 text-[var(--ds-accent)] font-semibold">
      {children}
    </span>
  );

  return (
    <div className="ua-lesson-stage">
      
      {/* Configuration Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[var(--ds-rule)] pb-4">
        <div>
          <h2 className="text-xl font-bold text-[var(--ds-ink)] tracking-tight">
            Frontier MoE Interactive System
          </h2>
          <p className="text-xs text-[var(--ds-faint)]">
            Explore gating functions, fine-grained active parameters, distributed bottlenecks, and student distillation.
          </p>
        </div>

        {/* Top Level Preset Controls */}
        <div className="flex items-center gap-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] p-2 rounded">
          <span className="text-[10px] text-[var(--ds-faint)] font-bold uppercase tracking-wider">Model Preset:</span>
          <div className="flex gap-1.5">
            {Object.keys(MOE_PRESETS).map((key) => {
              const isActive = selectedPreset === key;
              return (
                <button
                  key={key}
                  data-math-control
                  onClick={() => {
                    setSelectedPreset(key);
                    if (key === 'deepseekV3') {
                      setNumExpertsActive(8);
                      setNumSharedExperts(1);
                    } else if (key === 'llama4Maverick') {
                      setNumExpertsActive(1);
                      setNumSharedExperts(1);
                    } else {
                      setNumExpertsActive(8);
                      setNumSharedExperts(0);
                    }
                  }}
                  className={`ds-btn font-bold py-1 px-2.5 text-xs rounded transition-all duration-200 ${
                    isActive
                      ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)] border border-[var(--ds-accent)]'
                      : 'bg-transparent text-[var(--ds-faint)] border border-[var(--ds-rule)] hover:bg-[var(--ds-paper)]'
                  }`}
                >
                  {MOE_PRESETS[key].label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* HERO WIDGET: Expert Router Workbench */}
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 relative overflow-hidden">
        
        {/* Workbench Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--ds-rule)] pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-[var(--ds-ok)]" />
            <span className="text-xs font-bold tracking-wider uppercase text-[var(--ds-ink)]">
              Active Expert Router Workbench
            </span>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            {/* Play Pause Controls */}
            <div className="flex items-center gap-2 bg-[var(--ds-paper-2)] p-1 rounded border border-[var(--ds-rule)]">
              <button
                data-math-control
                onClick={() => setIsPlaying(!isPlaying)}
                className={`ds-btn font-bold py-0.5 px-2 text-xs transition-all ${
                  isPlaying ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'bg-transparent text-[var(--ds-faint)]'
                }`}
              >
                {isPlaying ? 'PAUSE' : 'PLAY'}
              </button>
              
              <select
                value={workbenchSpeed}
                onChange={(e) => setWorkbenchSpeed(Number(e.target.value))}
                className="bg-transparent text-xs font-semibold text-[var(--ds-faint)] outline-none border-l border-[var(--ds-rule)] pl-2 pr-1 cursor-pointer"
              >
                <option value={2500}>Slow (2.5s)</option>
                <option value={1500}>Normal (1.5s)</option>
                <option value={800}>Fast (0.8s)</option>
              </select>
            </div>

            {/* Prev / Next */}
            <div className="flex gap-1">
              <button
                data-math-control
                onClick={() => setTokenIndex(prev => (prev - 1 + STREAM_TOKENS.length) % STREAM_TOKENS.length)}
                className="ds-btn px-2 py-0.5 rounded bg-transparent border border-[var(--ds-rule)] text-[var(--ds-faint)] hover:bg-[var(--ds-paper-2)]"
              >
                &larr;
              </button>
              <button
                data-math-control
                onClick={() => setTokenIndex(prev => (prev + 1) % STREAM_TOKENS.length)}
                className="ds-btn px-2 py-0.5 rounded bg-transparent border border-[var(--ds-rule)] text-[var(--ds-faint)] hover:bg-[var(--ds-paper-2)]"
              >
                &rarr;
              </button>
            </div>
          </div>
        </div>

        {/* Workbench Core Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          
          {/* 1. Token Input Stream (3 cols) */}
          <div className="lg:col-span-3 flex flex-col gap-1.5 bg-[var(--ds-paper-2)] p-3 border border-[var(--ds-rule)]">
            <div className="text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-1 flex items-center justify-between">
              <span>Input Tokens</span>
              <span className="text-[9px] text-[var(--ds-accent)]">t={tokenIndex}</span>
            </div>
            <div className="flex flex-wrap lg:flex-col gap-1 max-h-[240px] overflow-y-auto pr-1">
              {STREAM_TOKENS.map((tk, idx) => {
                const isActive = idx === tokenIndex;
                return (
                  <div
                    key={idx}
                    onClick={() => setTokenIndex(idx)}
                    className={`px-2 py-1.5 rounded text-xs font-semibold cursor-pointer transition-all duration-200 flex items-center justify-between border ${
                      isActive
                        ? 'border-[var(--ds-accent)] bg-[var(--ds-accent-w)] text-[var(--ds-accent)] font-bold scale-105 shadow-sm'
                        : 'border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-faint)] hover:border-[var(--ds-ink)]'
                    }`}
                  >
                    <span>&ldquo;{tk.text}&rdquo;</span>
                    <span className="text-[9px] px-1 py-0.25 rounded border font-mono capitalize opacity-80">
                      {tk.domain}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 2. Routing Visualization Pipeline (7 cols) */}
          <div className="lg:col-span-7 relative flex flex-col items-center bg-slate-100/50 p-4 border border-[var(--ds-rule)] min-h-[220px] justify-center overflow-x-auto">
            
            <div className="w-full flex flex-row items-center justify-between gap-2.5 relative min-w-[420px]">
              
              {/* Active Token */}
              <div className="flex flex-col items-center shrink-0">
                <div className="mb-1.5 text-[9px] uppercase font-bold text-[var(--ds-mute)]">Router Input</div>
                <div className={`px-2.5 py-1.5 rounded border-2 font-mono text-xs font-bold shadow-sm transition-all duration-300 ${activeTokenObj.color}`}>
                  x = &ldquo;{activeTokenObj.text}&rdquo;
                </div>
              </div>

              {/* Arrow */}
              <div className="flex flex-col items-center text-[var(--ds-mute)]">
                <div className="text-[8px] font-mono text-[var(--ds-accent)]">W_gate * x</div>
                <ArrowRight className="w-4 h-4 text-[var(--ds-rule)]" />
              </div>

              {/* Router Gating Module */}
              <div className="flex flex-col items-center shrink-0 bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-2.5 w-32">
                <div className="text-[9px] uppercase font-bold text-[var(--ds-mute)] mb-2">Gating Logits</div>
                
                <div className="w-full space-y-1.5">
                  {['Math', 'Code', 'Common', 'Legal'].map((cat, i) => {
                    const domainMatch = activeTokenObj.domain === cat.toLowerCase();
                    const val = domainMatch ? 0.88 : (i === 2 ? 0.08 : 0.02);
                    return (
                      <div key={cat} className="space-y-0.5">
                        <div className="flex justify-between text-[8px] text-[var(--ds-faint)] font-mono">
                          <span>{cat}</span>
                          <span className={domainMatch ? 'text-[var(--ds-accent)] font-bold' : ''}>{(val * 100).toFixed(0)}%</span>
                        </div>
                        <span className="w-full block bg-slate-200/80 h-1.5 rounded overflow-hidden">
                          <span
                            className="h-full block transition-all duration-500"
                            style={{
                              width: `${val * 100}%`,
                              backgroundColor: domainMatch ? 'var(--ds-accent)' : 'var(--ds-mute)'
                            }}
                          />
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Arrow */}
              <div className="flex flex-col items-center text-[var(--ds-mute)]">
                <div className="text-[8px] font-mono text-[var(--ds-accent)]">Top-k + Shared</div>
                <ArrowRight className="w-4 h-4 text-[var(--ds-rule)]" />
              </div>

              {/* Target Experts Processing */}
              <div className="flex flex-col gap-1.5 shrink-0">
                <div className="text-[9px] uppercase font-bold text-[var(--ds-mute)] text-center">Active Experts</div>
                
                {numSharedExperts > 0 && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-yellow-600/30 bg-yellow-50 text-yellow-950 text-[10px] font-bold">
                    <Layers className="w-3.5 h-3.5" />
                    <span>Shared Expert</span>
                  </span>
                )}

                <span className="flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-[var(--ds-accent)] bg-[var(--ds-accent-w)] text-[var(--ds-accent)] text-[10px] font-bold animate-pulse">
                  <Cpu className="w-3.5 h-3.5 animate-spin" />
                  <span className="capitalize">{activeTokenObj.domain} Expert</span>
                </span>
              </div>

            </div>

            {/* Explanation */}
            <div className="mt-4 text-center text-xs text-[var(--ds-faint)] bg-[var(--ds-panel)] p-2.5 border border-[var(--ds-rule)] w-full max-w-lg">
              Token <span className="font-mono text-[var(--ds-ink)] font-bold">&ldquo;{activeTokenObj.text}&rdquo;</span> is dispatched to the 
              <span className="text-[var(--ds-ink)] font-bold capitalize"> {activeTokenObj.domain} Expert</span> (probability 
              <span className="font-mono text-[var(--ds-accent)] font-bold"> 88%</span>)
              {numSharedExperts > 0 && ' and simultaneously processes through the dense Shared Expert block.'}
            </div>

          </div>

          {/* 3. Live KPI Dashboard (2 cols) */}
          <div className="lg:col-span-2 grid grid-cols-2 lg:grid-cols-1 gap-2">
            <div className="bg-[var(--ds-panel)] p-3 border border-[var(--ds-rule)] flex flex-col justify-between">
              <div className="flex items-center justify-between text-[var(--ds-mute)] mb-0.5">
                <span className="text-[10px] font-bold uppercase">Active FLOPs</span>
                <Activity className="w-3.5 h-3.5 text-[var(--ds-accent)]" />
              </div>
              <div className="text-xl font-bold font-mono tracking-tight text-[var(--ds-accent)]">{kpiMetrics.flopsUtilized}</div>
              <div className="text-[9px] text-[var(--ds-faint)] mt-0.5">Compute saved vs. dense</div>
            </div>

            <div className="bg-[var(--ds-panel)] p-3 border border-[var(--ds-rule)] flex flex-col justify-between">
              <div className="flex items-center justify-between text-[var(--ds-mute)] mb-0.5">
                <span className="text-[10px] font-bold uppercase">Straggler Delay</span>
                <Gauge className="w-3.5 h-3.5 text-[var(--ds-accent)]" />
              </div>
              <div className="text-xl font-bold font-mono tracking-tight text-[var(--ds-accent)]">{kpiMetrics.stragglerDelay}</div>
              <div className="text-[9px] text-[var(--ds-faint)] mt-0.5">GPU synchronization cost</div>
            </div>

            <div className="bg-[var(--ds-panel)] p-3 border border-[var(--ds-rule)] flex flex-col justify-between">
              <div className="flex items-center justify-between text-[var(--ds-mute)] mb-0.5">
                <span className="text-[10px] font-bold uppercase">Token Drop</span>
                <AlertTriangle className="w-3.5 h-3.5 text-[var(--ds-warm)]" />
              </div>
              <div className={`text-xl font-bold font-mono tracking-tight ${
                parseFloat(kpiMetrics.droppingRate) > 0 ? 'text-[var(--ds-warm)] font-extrabold' : 'text-[var(--ds-mute)]'
              }`}>{kpiMetrics.droppingRate}</div>
              <div className="text-[9px] text-[var(--ds-faint)] mt-0.5">Buffer capacity overflows</div>
            </div>

            <div className="bg-[var(--ds-panel)] p-3 border border-[var(--ds-rule)] flex flex-col justify-between">
              <div className="flex items-center justify-between text-[var(--ds-mute)] mb-0.5">
                <span className="text-[10px] font-bold uppercase">All-to-All Dispatch</span>
                <TrendingUp className="w-3.5 h-3.5 text-[var(--ds-accent)]" />
              </div>
              <div className="text-xl font-bold font-mono tracking-tight text-[var(--ds-accent)]">{kpiMetrics.dispatchSize}</div>
              <div className="text-[9px] text-[var(--ds-faint)] mt-0.5">Cross-GPU network transfer</div>
            </div>
          </div>

        </div>
      </div>

      {/* MAIN LAYOUT: Tabs + Interactive Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Hand Navigation (3 cols) */}
        <div className="lg:col-span-3 bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-2.5 sticky top-[80px] z-30">
          <div className="text-[10px] font-bold uppercase text-[var(--ds-mute)] tracking-wider px-2 mb-2">
            Concepts
          </div>
          <div className="flex flex-col gap-1">
            {TABS.map((t) => {
              const TabIcon = t.icon;
              const isActive = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  data-math-control
                  onClick={() => setActiveTab(t.id)}
                  className={`ds-btn w-full flex items-center gap-3 px-3 py-2 text-xs font-bold text-left rounded transition-all duration-150 ${
                    isActive
                      ? 'bg-[var(--ds-accent-w)] text-[var(--ds-accent)] border border-[var(--ds-accent)] border-l-4'
                      : 'bg-transparent text-[var(--ds-faint)] border border-[var(--ds-rule)] hover:bg-[var(--ds-paper-2)] hover:text-[var(--ds-ink)] border-l-2'
                  }`}
                >
                  <TabIcon className="w-4 h-4 shrink-0" />
                  <span>{t.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Interactive Panel Content (9 cols) */}
        <div className="lg:col-span-9 bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-6 min-h-[480px]">
          
          {/* TAB: Why Sparse MoE */}
          {activeTab === 'why-moe' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Zap className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Why Sparse Mixture-of-Experts?</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                  <p>
                    In the scaling race, expanding model capacity is essential for reasoning depth. However, standard dense transformers require that every training and inference token compute on all parameters, raising FLOP bounds linearly.
                  </p>
                  <p className="font-semibold text-[var(--ds-ink)]">
                    Sparse MoE bypasses this by routing inputs only to highly-specialized blocks.
                  </p>
                  <div className="border border-[var(--ds-accent)] bg-[var(--ds-accent-w)] p-3.5 space-y-1.5">
                    <span className="font-bold block uppercase text-[9px] text-[var(--ds-accent)]">The Sparse Equation:</span>
                    <p className="text-[12px] text-[var(--ds-faint)] leading-normal">
                      Tokens dynamically select expert pathways, and outputs are blended according to the router logits:
                    </p>
                    <div className="text-center py-1.5 font-mono text-sm text-[var(--ds-accent)] font-bold">
                      y = &sum;<sub>i &isin; TopK</sub> G<sub>i</sub>(x) &middot; E<sub>i</sub>(x)
                    </div>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-5 border border-[var(--ds-rule)] flex flex-col justify-between">
                  <div>
                    <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-4">
                      Capacity vs Active Compute (FLOPs)
                    </h4>
                    <div className="space-y-4">
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-[var(--ds-faint)]">Total Parameters:</span>
                          <span className="font-mono text-[var(--ds-ink)] font-bold">{preset.totalParamsB}B</span>
                        </div>
                        <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                          <span className="bg-[var(--ds-accent)] h-full block" style={{ width: '100%' }} />
                        </span>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-[var(--ds-faint)]">Active parameters (per token):</span>
                          <span className="font-mono text-[var(--ds-accent)] font-bold">{preset.activeParamsB}B</span>
                        </div>
                        <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                          <span className="bg-[var(--ds-accent)] h-full block animate-pulse" style={{ width: `${(preset.activeParamsB / preset.totalParamsB) * 100}%` }} />
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="text-[11px] text-[var(--ds-mute)] mt-4 leading-normal">
                    <strong>Takeaway:</strong> Active parameter compute maps to inference latency (speed), while total parameters store multi-task capabilities without drawing active FLOP overhead.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Dense vs Sparse */}
          {activeTab === 'dense-vs-moe' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Layers className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Dense vs Sparse Layer Forward Passes</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Dense FFN Panel */}
                <div className="bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] p-4">
                  <h4 className="text-xs font-bold text-[var(--ds-faint)] uppercase tracking-wider mb-3 flex items-center gap-2">
                    <span className="w-2 h-2 bg-[var(--ds-warm)] rounded-full" />
                    Standard Dense MLP Block
                  </h4>
                  <p className="text-xs text-[var(--ds-faint)] mb-4">
                    Every token activates 100% of the MLP parameter weights.
                  </p>
                  
                  <div className="space-y-2 py-2">
                    {[1, 2, 3].map((tk) => (
                      <div key={tk} className="flex items-center gap-2.5 bg-[var(--ds-panel)] p-2 border border-[var(--ds-rule)]">
                        <span className="text-[10px] font-mono text-[var(--ds-warm)] bg-[var(--ds-warm-w)] px-1.5 py-0.5 rounded border border-[var(--ds-rule)]">Token {tk}</span>
                        <span className="text-[var(--ds-mute)]">&rarr;</span>
                        <div className="text-[10px] font-mono text-[var(--ds-ink)] px-2 py-1 rounded w-full text-center border border-[var(--ds-rule)]">
                          Full MLP parameters activated
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sparse MoE Panel */}
                <div className="bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] p-4">
                  <h4 className="text-xs font-bold text-[var(--ds-faint)] uppercase tracking-wider mb-3 flex items-center gap-2">
                    <span className="w-2 h-2 bg-[var(--ds-accent)] rounded-full" />
                    Sparse MoE Block (Top-1)
                  </h4>
                  <p className="text-xs text-[var(--ds-faint)] mb-4">
                    A gating router selects a single, specialized expert module for each token.
                  </p>

                  <div className="space-y-2 py-2">
                    {[
                      { id: 1, exp: 'Math Expert', color: 'text-purple-950 border-purple-300 bg-purple-50' },
                      { id: 2, exp: 'Code Expert', color: 'text-emerald-950 border-emerald-300 bg-emerald-50' },
                      { id: 3, exp: 'Translation Expert', color: 'text-orange-950 border-orange-300 bg-orange-50' }
                    ].map((tk) => (
                      <div key={tk.id} className="flex items-center gap-2.5 bg-[var(--ds-panel)] p-2 border border-[var(--ds-rule)]">
                        <span className="text-[10px] font-mono text-[var(--ds-accent)] bg-[var(--ds-accent-w)] px-1.5 py-0.5 rounded border border-[var(--ds-rule)]">Token {tk.id}</span>
                        <span className="text-[var(--ds-mute)]">&rarr;</span>
                        <div className={`text-[10px] font-mono px-2 py-1 rounded w-full text-center border font-bold ${tk.color}`}>
                          {tk.exp} (Sparse active params)
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Parameter Budget */}
          {activeTab === 'active-vs-total' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Database className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Dynamic Parameter Budget Calculator</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-5">
                  <div>
                    <label className="block text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Routed Experts Activated (Top-K): {numExpertsActive}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="16"
                      value={numExpertsActive}
                      onChange={(e) => setNumExpertsActive(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Shared Experts Activated: {numSharedExperts}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="4"
                      value={numSharedExperts}
                      onChange={(e) => setNumSharedExperts(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div className="p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)]">
                    <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase block">Active Formula:</span>
                    <p className="font-mono text-[10px] text-[var(--ds-faint)] mt-1">
                      Params = Backbone + (K * expert_size) + (Shared * expert_size)
                    </p>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] flex flex-col justify-between">
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Distribution Details
                    </h4>
                    
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-[var(--ds-faint)]">Dense Backbone:</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{preset.denseBackboneB}B</span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-[var(--ds-faint)]">Routed Experts ({numExpertsActive} active):</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{(numExpertsActive * preset.expertSizeB).toFixed(1)}B</span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-[var(--ds-faint)]">Shared Experts ({numSharedExperts} active):</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{(numSharedExperts * preset.expertSizeB).toFixed(1)}B</span>
                    </div>

                    <div className="border-t border-[var(--ds-rule)] pt-2.5 flex justify-between items-center text-sm font-bold">
                      <span className="text-[var(--ds-accent)]">Active Parameter Size:</span>
                      <span className="font-mono text-[var(--ds-accent)] text-md">
                        {(preset.denseBackboneB + (numExpertsActive * preset.expertSizeB) + (numSharedExperts * preset.expertSizeB)).toFixed(1)}B
                      </span>
                    </div>
                  </div>

                  <p className="text-[10px] text-[var(--ds-mute)] mt-4">
                    <strong>Notice:</strong> Increasing Top-K improves semantic precision but spikes serving latency due to GPU execution sync points.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Routing Gating */}
          {activeTab === 'top-k-routing' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <ShuffleIcon className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Top-K Routing Mechanics</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                    Logits are computed via vector projection <MathBlock>W_gate * x</MathBlock>. Noise can be added to encourage exploration and prevent dead experts.
                  </p>
                  
                  <div className="space-y-4 border border-[var(--ds-rule)] p-4 bg-[var(--ds-paper-2)]">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Routing Temperature:</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{routingTemp}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="2.0"
                        step="0.1"
                        value={routingTemp}
                        onChange={(e) => setRoutingTemp(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Noisy Gating Sigma (&epsilon;):</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{routingNoise}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="0.5"
                        step="0.05"
                        value={routingNoise}
                        onChange={(e) => setRoutingNoise(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)]">
                  <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-4">
                    Probabilities Over N Experts
                  </h4>

                  <div className="space-y-3">
                    {[
                      { name: 'Expert 1 (Math specialist)', base: 0.75 },
                      { name: 'Expert 2 (Generalist)', base: 0.18 },
                      { name: 'Expert 3 (Coder)', base: 0.05 },
                      { name: 'Expert 4 (Inactive)', base: 0.02 }
                    ].map((item) => {
                      const expVal = Math.exp((item.base + (Math.random() - 0.5) * routingNoise) / routingTemp);
                      return { name: item.name, expVal };
                    }).map((item, idx, arr) => {
                      const sum = arr.reduce((acc, curr) => acc + curr.expVal, 0);
                      const finalVal = item.expVal / sum;
                      const isTop1 = idx === 0;

                      return (
                        <div key={item.name} className="space-y-1">
                          <div className="flex justify-between text-[11px] font-mono">
                            <span className={isTop1 ? 'text-[var(--ds-accent)] font-bold' : 'text-[var(--ds-faint)]'}>{item.name}</span>
                            <span className={isTop1 ? 'text-[var(--ds-accent)] font-bold' : 'text-[var(--ds-faint)]'}>{(finalVal * 100).toFixed(1)}%</span>
                          </div>
                          <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                            <span
                              className="h-full block transition-all duration-300"
                              style={{
                                width: `${finalVal * 100}%`,
                                backgroundColor: isTop1 ? 'var(--ds-accent)' : 'var(--ds-mute)'
                              }}
                            />
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Shared vs Routed Experts */}
          {activeTab === 'shared-vs-routed' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <GitBranch className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Shared Experts vs Fine-grained Routed Experts</h3>
              </div>

              <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                <p>
                  Frontier architectures employ a hybrid route layout. Instead of making all experts sparse, we introduce small **Shared Experts** acting dense, combined with a larger pool of routed sparse experts.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                  <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] space-y-1.5">
                    <span className="text-xs font-bold text-[var(--ds-warm)] uppercase tracking-wide block">Shared Experts (Dense)</span>
                    <p className="text-xs text-[var(--ds-faint)]">
                      Compute on all tokens. They absorb global common knowledge (grammar, punctuation, common sentence logic), offloading basic parameters so routed experts can specialize deeply on niche fields.
                    </p>
                  </div>

                  <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] space-y-1.5">
                    <span className="text-xs font-bold text-[var(--ds-accent)] uppercase tracking-wide block">Routed Experts (Sparse)</span>
                    <p className="text-xs text-[var(--ds-faint)]">
                      Only compute when assigned by the router. Because they are relieved of global memory burdens, they specialize in complex target concepts (calculus formulas, programming syntax, specific translations).
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Load Balancing */}
          {activeTab === 'load-balancing' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <ScaleIcon className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Load Balancing & Gating Collapse</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                    Without load balancing pressure, the router collapses and routes all tokens to the same 1-2 "popular" experts, starving the rest (creating dead experts). We counter this with an auxiliary loss:
                  </p>

                  <div className="p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] font-mono text-[11px] text-[var(--ds-accent)] text-center font-bold">
                    L_aux = &lambda; &middot; N &sum; f_i &middot; P_i
                  </div>

                  <div className="space-y-4 border border-[var(--ds-rule)] p-4 bg-[var(--ds-paper-2)]">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Auxiliary Loss Weight (&lambda;):</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{auxLossWeight}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="0.1"
                        step="0.01"
                        value={auxLossWeight}
                        onChange={(e) => setAuxLossWeight(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Expert Capacity Factor:</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{capacityFactor}x</span>
                      </div>
                      <input
                        type="range"
                        min="0.8"
                        max="2.0"
                        step="0.1"
                        value={capacityFactor}
                        onChange={(e) => setCapacityFactor(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] flex flex-col justify-between">
                  <div>
                    <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-4">
                      Expert Load Distribution
                    </h4>

                    <div className="space-y-3">
                      {[
                        { name: 'Expert A (Math)', load: auxLossWeight < 0.02 ? 96 : 74 },
                        { name: 'Expert B (Code)', load: auxLossWeight < 0.02 ? 88 : 71 },
                        { name: 'Expert C (Common)', load: auxLossWeight < 0.02 ? 22 : 65 },
                        { name: 'Expert D (Legal)', load: auxLossWeight < 0.02 ? 4 : 61 }
                      ].map((exp) => {
                        const isOverflow = exp.load > (capacityFactor * 65);
                        return (
                          <div key={exp.name} className="space-y-1">
                            <div className="flex justify-between text-[11px] font-mono">
                              <span>{exp.name}</span>
                              <span className={isOverflow ? 'text-[var(--ds-warm)] font-bold' : 'text-[var(--ds-faint)]'}>
                                {exp.load}% {isOverflow && '(Over capacity!)'}
                              </span>
                            </div>
                            <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                              <span
                                className="h-full block transition-all duration-300"
                                style={{
                                  width: `${exp.load}%`,
                                  backgroundColor: isOverflow ? 'var(--ds-warm)' : 'var(--ds-ok)'
                                }}
                              />
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <p className="text-[10px] text-[var(--ds-mute)] mt-4">
                    <strong>Diagnostic:</strong> {auxLossWeight < 0.02 
                      ? 'Router collapse: all tokens hit Expert A/B. Rest of experts are dead.'
                      : 'Uniform routing: Workload balanced. Capacity threshold is stable.'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Expert Parallelism */}
          {activeTab === 'expert-parallelism' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Server className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Expert Parallelism & GPU Dispatch</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                    At scale, a model's experts are partition-hosted across multiple GPUs (Expert Parallelism). Shuffling tokens between local attention layers and remote experts requires two costly **All-to-All** communication passes.
                  </p>

                  <div className="space-y-2 border border-[var(--ds-rule)] p-4 bg-[var(--ds-paper-2)]">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-[var(--ds-faint)]">Inter-GPU Network Latency:</span>
                      <span className="font-mono font-bold text-[var(--ds-accent)]">{networkLatency} ms</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      value={networkLatency}
                      onChange={(e) => setNetworkLatency(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)]">
                  <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-3">
                    Distributed Parallel Execution Cycle
                  </h4>

                  <div className="space-y-2.5 font-mono text-[10px] text-[var(--ds-faint)]">
                    <div className="flex justify-between border-b border-slate-200 pb-1.5">
                      <span>1. Local QKV calculations</span>
                      <span className="text-[var(--ds-ok)] font-bold">Done</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-200 pb-1.5">
                      <span>2. Gating router logits mapping</span>
                      <span className="text-[var(--ds-ok)] font-bold">Done</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-200 pb-1.5">
                      <span>3. All-to-All Token Dispatch</span>
                      <span className={networkLatency > 40 ? 'text-[var(--ds-warm)] font-bold' : 'text-[var(--ds-accent)] font-bold'}>
                        {networkLatency > 40 ? 'Network Bottleneck' : 'Active'} ({networkLatency}ms)
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-slate-200 pb-1.5">
                      <span>4. Remote Expert matrix multiplication</span>
                      <span className="text-[var(--ds-ok)] font-bold">Active</span>
                    </div>
                    <div className="flex justify-between pb-1.5">
                      <span>5. All-to-All Expert Gather</span>
                      <span className="text-[var(--ds-mute)]">Pending</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: Failure Debugger */}
          {activeTab === 'failure-modes' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <AlertTriangle className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Frontier MoE Failure Mode Debugger</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {FAILURE_MODES.map((mode) => {
                  const isFixed = debuggerFixes[mode.id];
                  return (
                    <div
                      key={mode.id}
                      className={`p-4 border transition-all duration-300 flex flex-col justify-between ${
                        isFixed
                          ? 'border-[var(--ds-ok)] bg-emerald-50/30'
                          : 'border-[var(--ds-warm)] bg-rose-50/10'
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`w-2.5 h-2.5 rounded-full ${isFixed ? 'bg-[var(--ds-ok)]' : 'bg-[var(--ds-warm)] animate-pulse'}`} />
                          <h4 className="text-sm font-bold text-[var(--ds-ink)] capitalize">{mode.label}</h4>
                        </div>
                        
                        <div className="space-y-2 text-xs text-[var(--ds-faint)] mb-4 leading-normal">
                          <div><strong>Symptom:</strong> {mode.symptom}</div>
                          <div><strong>Fix Strategy:</strong> {mode.fix}</div>
                        </div>
                      </div>

                      <button
                        data-math-control
                        onClick={() => setDebuggerFixes(prev => ({ ...prev, [mode.id]: !prev[mode.id] }))}
                        className={`ds-btn font-bold py-1.5 px-3 rounded text-xs transition-all duration-200 ${
                          isFixed
                            ? 'bg-[var(--ds-ok)] text-white border border-[var(--ds-ok)]'
                            : 'bg-transparent text-[var(--ds-warm)] border border-[var(--ds-warm)] hover:bg-[var(--ds-warm-w)]'
                        }`}
                      >
                        {isFixed ? 'FIX APPLIED ✓' : 'APPLY ARCHITECTURAL FIX'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB: Expert Specialization */}
          {activeTab === 'expert-specialization' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Brain className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Expert Specialization & Semantic Maps</h3>
              </div>

              <div className="space-y-4">
                <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                  During scale pretraining, experts organically divide mathematical syntax, coding formats, grammatical patterns, and terminology.
                </p>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {TOKEN_DOMAINS.map((domain) => (
                    <div
                      key={domain.token}
                      className="p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[9px] uppercase font-bold text-[var(--ds-mute)]">Domain</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded capitalize bg-[var(--ds-panel)] text-[var(--ds-accent)] border border-[var(--ds-rule)]">
                          {domain.domain}
                        </span>
                      </div>
                      
                      <div className="text-center font-mono font-bold text-[var(--ds-ink)] text-sm my-2">
                        &ldquo;{domain.token}&rdquo;
                      </div>

                      <p className="text-[9px] text-[var(--ds-faint)] leading-tight">
                        Routes with high probability to the dedicated {domain.domain} expert.
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB: MoE Distillation */}
          {activeTab === 'distillation' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Sparkles className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">MoE Distillation & Student Routing</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                    Student routers are trained to align directly with teacher soft targets and gating probability maps using KL Divergence loss, retaining teacher intelligence at edge-deployable latency scales.
                  </p>

                  <div className="space-y-4 border border-[var(--ds-rule)] p-4 bg-[var(--ds-paper-2)]">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Soft Target Weight (&alpha;):</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{teacherSoftWeight}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.1"
                        value={teacherSoftWeight}
                        onChange={(e) => setTeacherSoftWeight(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Router Distillation Weight:</span>
                        <span className="font-mono font-bold text-[var(--ds-accent)]">{routerDistillWeight}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.1"
                        value={routerDistillWeight}
                        onChange={(e) => setRouterDistillWeight(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                  </div>

                  <button
                    data-math-control
                    onClick={handleRunDistillation}
                    disabled={isDistilling}
                    className="ds-btn primary w-full py-2 text-xs tracking-wider justify-center gap-2 rounded"
                  >
                    {isDistilling ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>DISTILLING MODEL ({distillProgress}%)</span>
                      </>
                    ) : (
                      <>
                        <Flame className="w-3.5 h-3.5" />
                        <span>RUN DISTILLATION PIPELINE</span>
                      </>
                    )}
                  </button>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] flex flex-col justify-between">
                  <div className="space-y-4">
                    <h4 className="text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Distillation Metrics
                    </h4>

                    <div>
                      <div className="flex justify-between text-xs text-[var(--ds-faint)] font-mono mb-1">
                        <span>Student Loss:</span>
                        <span className="text-[var(--ds-accent)] font-bold">{distillMetrics.studentLoss}</span>
                      </div>
                      <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                        <span
                          className="bg-[var(--ds-accent)] h-full block transition-all duration-350"
                          style={{ width: `${Math.min(100, (distillMetrics.studentLoss / 2.8) * 100)}%` }}
                        />
                      </span>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs text-[var(--ds-faint)] font-mono mb-1">
                        <span>Router KL Divergence:</span>
                        <span className="text-[var(--ds-accent)] font-bold">{distillMetrics.routerKl}</span>
                      </div>
                      <span className="w-full block bg-slate-200 h-2 rounded overflow-hidden">
                        <span
                          className="bg-[var(--ds-accent)] h-full block transition-all duration-350"
                          style={{ width: `${Math.min(100, (distillMetrics.routerKl / 0.9) * 100)}%` }}
                        />
                      </span>
                    </div>
                  </div>

                  <p className="text-[10px] text-[var(--ds-mute)] mt-4">
                    <strong>Goal:</strong> Force the student gating distributions to mimic the teacher, avoiding trial-and-error student explorer decay.
                  </p>
                </div>
              </div>
            </div>
          )}

        </div>

      </div>

      {/* Paper Signals Section */}
      <div className="bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-5">
        <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2.5 mb-3">
          <Brain className="w-5 h-5 text-[var(--ds-accent)]" />
          <h3 className="text-sm font-bold text-[var(--ds-ink)] uppercase tracking-wider">
            How to read MoE details in frontier research papers
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PAPER_SIGNAL_CARDS.map((card) => (
            <div key={card.title} className="p-4 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] rounded">
              <h4 className="text-xs font-bold text-[var(--ds-accent)] uppercase tracking-wide mb-1">{card.title}</h4>
              <div className="text-xs text-[var(--ds-faint)] mb-2 font-medium">
                <span className="text-[9px] uppercase font-bold text-[var(--ds-mute)] block">Paper Signals</span>
                {card.signals}
              </div>
              <p className="text-xs text-[var(--ds-ink)] italic border-l-2 border-[var(--ds-accent)] pl-2">
                {card.interpretation}
              </p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
