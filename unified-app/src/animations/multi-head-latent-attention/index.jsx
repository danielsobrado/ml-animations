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
  BarChart2,
  HardDrive,
  Users,
  Play,
  Pause,
  BookOpen,
  HelpCircle
} from 'lucide-react';
import { ATTENTION_MODES, PAPER_ANCHORS } from './data';

const TABS = [
  { id: 'kv-problem', label: '1. KV Cache Problem', icon: Zap },
  { id: 'mha-layout', label: '2. MHA Layout', icon: Layers },
  { id: 'mqa-gqa', label: '3. MQA/GQA Sharing', icon: GitBranch },
  { id: 'mla-latent', label: '4. MLA Latent Cache', icon: Cpu },
  { id: 'memory-comparison', label: '5. Memory Comparison', icon: BarChart2 },
  { id: 'decode-bandwidth', label: '6. Decode Bandwidth', icon: Gauge },
  { id: 'quality-memory', label: '7. Quality vs Memory', icon: TrendingUp },
  { id: 'serving-playground', label: '8. Long-Context Playground', icon: Server },
  { id: 'paper-decoder', label: '9. Paper Decoder', icon: BookOpen },
];

const PRESETS = {
  llama3: {
    name: 'Llama 3 8B (GQA)',
    queryHeads: 32,
    kvHeads: 8,
    headDim: 128,
    layers: 32,
    latentDim: 128,
    bytesPerValue: 2,
  },
  deepseekV2: {
    name: 'DeepSeek-V2 (MLA)',
    queryHeads: 128,
    kvHeads: 1, // MLA effective KV sharing representation
    headDim: 128,
    layers: 60,
    latentDim: 128,
    bytesPerValue: 2,
  },
  deepseekV3: {
    name: 'DeepSeek-V3 (SOTA MLA)',
    queryHeads: 128,
    kvHeads: 1,
    headDim: 128,
    layers: 61,
    latentDim: 128,
    bytesPerValue: 1, // fp8
  },
  classicMHA: {
    name: 'Classic 7B (MHA)',
    queryHeads: 32,
    kvHeads: 32,
    headDim: 128,
    layers: 32,
    latentDim: 128,
    bytesPerValue: 2,
  }
};

const STREAM_WORDS = [
  'The', 'theorem', 'states', 'that', 'every', 'bounded', 'sequence', 'of',
  'real', 'numbers', 'has', 'a', 'convergent', 'subsequence,', 'which', 'is',
  'fundamental', 'to', 'analysis.'
];

export default function MultiHeadLatentAttention() {
  const [activeTab, setActiveTab] = useState('kv-problem');
  const [selectedPreset, setSelectedPreset] = useState('deepseekV3');

  // Architecture parameters
  const [architecture, setArchitecture] = useState('mla');
  const [contextLength, setContextLength] = useState(32768);
  const [layers, setLayers] = useState(32);
  const [queryHeads, setQueryHeads] = useState(32);
  const [kvHeads, setKvHeads] = useState(8);
  const [headDim, setHeadDim] = useState(128);
  const [bytesPerValue, setBytesPerValue] = useState(2); // 2: FP16, 1: FP8, 0.5: INT4
  const [latentDim, setLatentDim] = useState(128);

  // Serving playground parameters
  const [memoryLimitGB, setMemoryLimitGB] = useState(80);
  const [concurrentRequests, setConcurrentRequests] = useState(16);
  const [hardwareProfile, setHardwareProfile] = useState('bandwidth-limited');

  // Morph loop states
  const [tokenStep, setTokenStep] = useState(4);
  const [isPlayingMorph, setIsPlayingMorph] = useState(true);

  // Pareto Quality settings
  const [memoryBudget, setMemoryBudget] = useState(24);
  const [qualityTargetContext, setQualityTargetContext] = useState(32768);

  // Paper Decoder choices
  const [paperSelectedText, setPaperSelectedText] = useState('mla');
  const [paperAnswers, setPaperAnswers] = useState({ q1: '', q2: '', q3: '' });

  // Sync parameters to selected presets
  useEffect(() => {
    const p = PRESETS[selectedPreset];
    if (p) {
      setQueryHeads(p.queryHeads);
      setKvHeads(p.kvHeads);
      setHeadDim(p.headDim);
      setLayers(p.layers);
      setLatentDim(p.latentDim);
      setBytesPerValue(p.bytesPerValue);
      // Auto toggle architecture mapping
      if (selectedPreset === 'classicMHA') {
        setArchitecture('mha');
      } else if (selectedPreset === 'llama3') {
        setArchitecture('gqa');
      } else if (selectedPreset === 'deepseekV2' || selectedPreset === 'deepseekV3') {
        setArchitecture('mla');
      }
    }
  }, [selectedPreset]);

  // Morph animation tick
  useEffect(() => {
    if (!isPlayingMorph) return;
    const interval = setInterval(() => {
      setTokenStep((prev) => (prev >= STREAM_WORDS.length ? 4 : prev + 1));
    }, 1800);
    return () => clearInterval(interval);
  }, [isPlayingMorph]);

  // Calculations
  const calculatedMetrics = useMemo(() => {
    // 1. MHA size: layers * tokens * heads * headDim * 2(K,V) * bytes
    const mhaBytes = layers * contextLength * queryHeads * headDim * 2 * bytesPerValue;
    
    // 2. MQA size: layers * tokens * 1 * headDim * 2(K,V) * bytes
    const mqaBytes = layers * contextLength * 1 * headDim * 2 * bytesPerValue;

    // 3. GQA size: layers * tokens * kvHeads * headDim * 2(K,V) * bytes
    const gqaBytes = layers * contextLength * Math.min(kvHeads, queryHeads) * headDim * 2 * bytesPerValue;

    // 4. MLA size: layers * tokens * (latentDim + decoupled_rope_head_dim) * bytes
    // Decoupled rope projection is standard in DeepSeek MLA (e.g. 64d for queries and keys)
    // For simplicity of formula block, we use standard teaching approximation:
    // cache_MLA = layers * tokens * latentDim * bytes
    // Plus a decoupled positional vector of 64d (which is stored in cache for keys only).
    // Let's approximate it as: layers * tokens * (latentDim + 64) * bytes.
    const decoupledDim = 64;
    const mlaBytes = layers * contextLength * (latentDim + decoupledDim) * bytesPerValue;

    // Memory variables depending on active architecture
    let activeBytes = mhaBytes;
    let cacheRatio = 1.0;
    if (architecture === 'mqa') {
      activeBytes = mqaBytes;
      cacheRatio = mqaBytes / mhaBytes;
    } else if (architecture === 'gqa') {
      activeBytes = gqaBytes;
      cacheRatio = gqaBytes / mhaBytes;
    } else if (architecture === 'mla') {
      activeBytes = mlaBytes;
      cacheRatio = mlaBytes / mhaBytes;
    }

    const activeGB = activeBytes / (1024 * 1024 * 1024);
    const mhaGB = mhaBytes / (1024 * 1024 * 1024);
    const gqaGB = gqaBytes / (1024 * 1024 * 1024);
    const mlaGB = mlaBytes / (1024 * 1024 * 1024);

    // Read per decode step
    const readPerStepBytes = activeBytes / contextLength;
    const readPerStepMB = readPerStepBytes / (1024 * 1024);

    // Bandwidth Bottleneck Score (0-100)
    // Limited hardware profiles scale memory access speed vs FLOPS
    let memSpeedGbps = 2000; // H100 SXM (~3.35 TB/s or 3350 GB/s)
    if (hardwareProfile === 'bandwidth-limited') memSpeedGbps = 800; // RTX 4090 (~1000 GB/s)
    if (hardwareProfile === 'balanced') memSpeedGbps = 2000;
    if (hardwareProfile === 'compute-limited') memSpeedGbps = 4000;

    // Memory read latency proxy = readPerStepMB * concurrentRequests / memSpeedGbps (scaled to score)
    const rawTraffic = (readPerStepBytes * concurrentRequests) / 1e9; // GB transferred per step
    const busSaturation = Math.min(100, (rawTraffic / (memSpeedGbps / 100)) * 100);

    // Compute load
    // MLA requires query-time up-projection matrix multiplication:
    // H_q * headDim * latentDim * 2 operations per token per layer
    const mlaExtraFlops = layers * queryHeads * headDim * latentDim * 2;
    const baselineFlops = layers * queryHeads * headDim * 2; // standard dot product
    const flopsLoad = architecture === 'mla' ? mlaExtraFlops / 1e6 : baselineFlops / 1e6; // MFLOPs

    // Quality-risk index (approximate heuristic based on compression density)
    let qualityRisk = 0;
    if (architecture === 'mha') qualityRisk = 5;
    if (architecture === 'gqa') {
      // risk increases as kvHeads decreases
      const shareFactor = queryHeads / kvHeads;
      qualityRisk = Math.min(95, 10 + shareFactor * 2);
    }
    if (architecture === 'mqa') qualityRisk = 85;
    if (architecture === 'mla') {
      // MLA latent dim relative to full head representation
      const fullDim = queryHeads * headDim;
      const compressionRatio = latentDim / fullDim;
      qualityRisk = Math.min(90, Math.max(10, 30 - compressionRatio * 150));
    }

    return {
      mhaGB: mhaGB.toFixed(2),
      gqaGB: gqaGB.toFixed(2),
      mlaGB: mlaGB.toFixed(2),
      activeGB: activeGB.toFixed(2),
      cacheRatioPercent: (cacheRatio * 100).toFixed(1) + '%',
      readPerStepMB: readPerStepMB.toFixed(2),
      busSaturation: busSaturation.toFixed(1),
      flopsMflops: flopsLoad.toFixed(2),
      qualityRisk: Math.round(qualityRisk),
      totalRequestGb: (activeGB * concurrentRequests).toFixed(2),
      fitsBudget: (activeGB * concurrentRequests) <= memoryLimitGB,
    };
  }, [layers, contextLength, queryHeads, kvHeads, headDim, bytesPerValue, latentDim, architecture, concurrentRequests, hardwareProfile, memoryLimitGB]);

  return (
    <div className="ua-lesson-stage">
      
      {/* HEADER SECTION */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[var(--ds-rule)] pb-4">
        <div>
          <h2 className="text-xl font-bold text-[var(--ds-ink)] tracking-tight">
            Multi-head Latent Attention / Attention Compression
          </h2>
          <p className="text-xs text-[var(--ds-faint)] mt-1">
            Analyze the frontier LLM inference memory bottleneck: how MHA, MQA, GQA, and MLA trade off memory bandwidth, cache sizes, and query compute.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex items-center gap-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] p-2 rounded shadow-sm shrink-0">
          <span className="text-[10px] text-[var(--ds-faint)] font-bold uppercase tracking-wider">Model Preset:</span>
          <div className="flex gap-1.5">
            {Object.keys(PRESETS).map((key) => {
              const isActive = selectedPreset === key;
              return (
                <button
                  key={key}
                  data-math-control
                  onClick={() => setSelectedPreset(key)}
                  className={`ds-btn font-bold py-1 px-2.5 text-xs rounded transition-all duration-200 ${
                    isActive
                      ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)] border border-[var(--ds-accent)]'
                      : 'bg-transparent text-[var(--ds-faint)] border border-[var(--ds-rule)] hover:bg-[var(--ds-paper)]'
                  }`}
                >
                  {PRESETS[key].name}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* HERO HERO HERO: KV MEMORY MORPH PIPELINE */}
      <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 relative overflow-hidden rounded-sm">
        
        {/* Visualizer Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--ds-rule)] pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-[var(--ds-accent)] animate-pulse" />
            <span className="text-xs font-bold tracking-wider uppercase text-[var(--ds-ink)]">
              Hero Visualization: KV Memory Morph
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Play/Pause */}
            <div className="flex items-center gap-2 bg-[var(--ds-paper-2)] p-1 rounded border border-[var(--ds-rule)]">
              <button
                data-math-control
                onClick={() => setIsPlayingMorph(!isPlayingMorph)}
                className={`ds-btn font-bold py-0.5 px-2.5 text-[10px] transition-all rounded ${
                  isPlayingMorph ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)]' : 'bg-transparent text-[var(--ds-faint)]'
                }`}
              >
                {isPlayingMorph ? 'PAUSE TICK' : 'PLAY TICK'}
              </button>
              <button
                data-math-control
                onClick={() => setTokenStep(4)}
                className="ds-btn py-0.5 px-2 text-[10px] text-[var(--ds-faint)] hover:text-[var(--ds-ink)]"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            </div>

            {/* Architecture Selector for Quick Comparison */}
            <div className="flex gap-1 bg-[var(--ds-paper-2)] p-0.5 rounded border border-[var(--ds-rule)]">
              {Object.keys(ATTENTION_MODES).map((mode) => (
                <button
                  key={mode}
                  data-math-control
                  onClick={() => setArchitecture(mode)}
                  className={`ds-btn py-0.5 px-2 text-[10px] font-bold rounded ${
                    architecture === mode
                      ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                      : 'bg-transparent text-[var(--ds-faint)] hover:text-[var(--ds-ink)]'
                  }`}
                >
                  {mode.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Animation Core */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          
          {/* Sentence / Token view */}
          <div className="lg:col-span-4 space-y-3">
            <span className="text-[9px] uppercase font-bold text-[var(--ds-faint)] tracking-wider block">Sequence Generating Loop</span>
            
            {/* Tokens list */}
            <div className="flex flex-wrap gap-1 p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] rounded">
              {STREAM_WORDS.map((word, idx) => {
                const isGenerated = idx < tokenStep;
                const isNew = idx === tokenStep;
                return (
                  <span
                    key={idx}
                    className={`px-1.5 py-1 rounded text-xs transition-all duration-300 ${
                      isNew
                        ? 'border border-[var(--ds-accent)] bg-[var(--ds-accent-w)] text-[var(--ds-accent)] font-bold scale-105 shadow-sm animate-pulse'
                        : isGenerated
                        ? 'border border-[var(--ds-rule)] bg-[var(--ds-panel)] text-[var(--ds-ink)]'
                        : 'text-slate-300 bg-transparent border border-transparent select-none'
                    }`}
                  >
                    {word}
                  </span>
                );
              })}
            </div>

            {/* Micro instructions */}
            <div className="p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] rounded text-xs text-[var(--ds-faint)] space-y-1.5">
              <div>
                <span className="font-semibold text-[var(--ds-ink)]">At Step t = {tokenStep + 1}:</span>
                <ul className="list-disc pl-4 mt-1 space-y-1 text-[11px]">
                  <li>Compute Query <span className="font-mono text-blue-600">Q</span> for the new token.</li>
                  <li>Read KV memory cache of the <span className="font-bold text-[var(--ds-ink)]">{tokenStep} previous</span> tokens.</li>
                  <li>Append new token's key/value representations to the cache.</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Caching Lanes Graphics */}
          <div className="lg:col-span-8 bg-slate-100/40 p-4 border border-[var(--ds-rule)] rounded min-h-[220px] flex flex-col justify-between">
            <div className="flex justify-between items-center text-[10px] font-bold uppercase text-[var(--ds-faint)] mb-3">
              <span>Attention Cache Storage Geometry</span>
              <span className="text-[var(--ds-accent)] capitalize">{ATTENTION_MODES[architecture].label}</span>
            </div>

            {/* Layout grid simulating cached heads per word */}
            <div className="space-y-2 max-h-[160px] overflow-y-auto pr-1">
              {Array.from({ length: Math.min(6, tokenStep) }).map((_, wordIdx) => {
                const word = STREAM_WORDS[wordIdx];
                return (
                  <div key={wordIdx} className="flex items-center gap-3 bg-[var(--ds-panel)] p-2 rounded border border-[var(--ds-rule)] shadow-xs">
                    {/* Word Label */}
                    <span className="w-16 text-center text-[10px] font-bold text-[var(--ds-faint)] border-r border-[var(--ds-rule)] pr-2 shrink-0 truncate">
                      &ldquo;{word}&rdquo;
                    </span>

                    {/* Heads grid */}
                    <div className="flex flex-wrap gap-1.5 items-center w-full">
                      {architecture === 'mha' && (
                        <>
                          {/* MHA: show separate K/V boxes per head */}
                          {Array.from({ length: 8 }).map((_, headIdx) => (
                            <span key={headIdx} className="flex gap-0.5 shrink-0 animate-fade-in">
                              <span className="w-4 h-4 rounded-xs bg-emerald-500 border border-emerald-600 text-[8px] text-white flex items-center justify-center font-mono font-bold shadow-xs">K{headIdx+1}</span>
                              <span className="w-4 h-4 rounded-xs bg-amber-500 border border-amber-600 text-[8px] text-white flex items-center justify-center font-mono font-bold shadow-xs">V{headIdx+1}</span>
                            </span>
                          ))}
                          <span className="text-[8.5px] text-[var(--ds-faint)] italic ml-2">(8 separate K/V heads stored)</span>
                        </>
                      )}

                      {architecture === 'mqa' && (
                        <>
                          {/* MQA: show only one shared K/V box */}
                          <span className="flex gap-0.5 shrink-0 animate-fade-in">
                            <span className="w-6 h-6 rounded-xs bg-emerald-600 border border-emerald-700 text-[10px] text-white flex items-center justify-center font-mono font-bold shadow-xs">K</span>
                            <span className="w-6 h-6 rounded-xs bg-amber-600 border border-amber-700 text-[10px] text-white flex items-center justify-center font-mono font-bold shadow-xs">V</span>
                          </span>
                          <span className="text-[8.5px] text-[var(--ds-faint)] font-bold ml-2">All 32 query heads read this single shared head pair.</span>
                        </>
                      )}

                      {architecture === 'gqa' && (
                        <>
                          {/* GQA: show grouped shared boxes */}
                          {Array.from({ length: 2 }).map((_, groupIdx) => (
                            <span key={groupIdx} className="flex items-center gap-0.5 border border-[var(--ds-rule)] p-0.5 bg-[var(--ds-paper-2)] shrink-0 animate-fade-in">
                              <span className="w-4 h-4 rounded-xs bg-emerald-500 border border-emerald-600 text-[8px] text-white flex items-center justify-center font-mono font-bold">K{groupIdx+1}</span>
                              <span className="w-4 h-4 rounded-xs bg-amber-500 border border-amber-600 text-[8px] text-white flex items-center justify-center font-mono font-bold">V{groupIdx+1}</span>
                              <span className="text-[8px] text-[var(--ds-faint)] px-1 font-mono uppercase font-bold">Group{groupIdx+1}</span>
                            </span>
                          ))}
                          <span className="text-[8.5px] text-[var(--ds-faint)] italic ml-2">(Query groups share 2 cached KV pairs)</span>
                        </>
                      )}

                      {architecture === 'mla' && (
                        <>
                          {/* MLA: show compressed latent capsule */}
                          <span className="flex items-center gap-1.5 px-3 py-1 rounded bg-purple-100 border border-purple-300 text-purple-950 text-[10px] font-bold shrink-0 animate-fade-in shadow-xs">
                            <Cpu className="w-3.5 h-3.5 text-purple-700 animate-pulse" />
                            <span>Compressed Latent Vector c_t ({latentDim}d)</span>
                          </span>

                          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-100 border border-emerald-300 text-emerald-950 text-[9px] font-bold shrink-0 animate-fade-in">
                            <span>Positional Key Vector k_R (64d)</span>
                          </span>

                          {/* Spin projection core indicator */}
                          <span className="flex items-center gap-1 text-[8.5px] text-[var(--ds-accent)] font-semibold shrink-0">
                            <RotateCcw className="w-2.5 h-2.5 animate-spin" />
                            <span>Query-time up-project (Absorbed)</span>
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Live Metrics readout */}
            <div className="mt-3 grid grid-cols-3 gap-2 border-t border-[var(--ds-rule)] pt-3">
              <div className="text-center">
                <span className="text-[9px] uppercase font-bold text-[var(--ds-faint)] block">Cache Footprint</span>
                <span className="text-sm font-bold text-[var(--ds-accent)] font-mono">{calculatedMetrics.activeGB} GB</span>
              </div>
              <div className="text-center">
                <span className="text-[9px] uppercase font-bold text-[var(--ds-faint)] block">Traffic / Token</span>
                <span className="text-sm font-bold text-[var(--ds-accent)] font-mono">{calculatedMetrics.readPerStepMB} MB</span>
              </div>
              <div className="text-center">
                <span className="text-[9px] uppercase font-bold text-[var(--ds-faint)] block">Cache vs MHA</span>
                <span className="text-sm font-bold text-[var(--ds-accent)] font-mono">{calculatedMetrics.cacheRatioPercent}</span>
              </div>
            </div>

          </div>

        </div>
      </div>

      {/* DETAILED LESSON WORKBENCH */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Sidebar Nav (3 cols) */}
        <div className="lg:col-span-3 bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-2.5 sticky top-[80px] z-30 rounded-sm">
          <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider px-2 mb-2 block">
            Lesson Sections
          </span>
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

          <div className="mt-4 p-3 bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] rounded-sm space-y-2">
            <span className="text-[9px] uppercase font-bold text-[var(--ds-faint)] block">Aesthetic Token Config</span>
            <div className="space-y-1">
              <label className="text-[10px] text-[var(--ds-faint)] font-mono block">Context Tokens: {contextLength}</label>
              <input
                type="range"
                min="4096"
                max="128000"
                step="4096"
                value={contextLength}
                onChange={(e) => setContextLength(Number(e.target.value))}
                className="ds-range"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-[var(--ds-faint)] font-mono block">Model Layers: {layers}</label>
              <input
                type="range"
                min="12"
                max="80"
                step="4"
                value={layers}
                onChange={(e) => setLayers(Number(e.target.value))}
                className="ds-range"
              />
            </div>
          </div>
        </div>

        {/* Interactive content (9 cols) */}
        <div className="lg:col-span-9 bg-[var(--ds-panel)] border border-[var(--ds-rule)] p-6 min-h-[480px] rounded-sm">
          
          {/* TAB 1: THE KV CACHE PROBLEM */}
          {activeTab === 'kv-problem' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Zap className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">The KV Cache Inference Bottleneck</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                  <p>
                    During LLM generation (autoregressive decoding), the model outputs text one token at a time. To calculate attention scores, the new token must attend to all previous tokens.
                  </p>
                  <p className="font-semibold text-[var(--ds-ink)]">
                    Recomputing key/value projections for all old tokens at every step is mathematically wasteful.
                  </p>
                  <p>
                    So, we cache them. But as sequences grow, this cache dominates GPU memory. Let's calculate its bytes footprint manually:
                  </p>

                  <div className="border border-[var(--ds-accent)] bg-[var(--ds-accent-w)] p-3 rounded space-y-1.5">
                    <span className="font-bold block uppercase text-[9px] text-[var(--ds-accent)]">KV Cache Equation:</span>
                    <div className="font-mono text-center py-2 text-[13px] text-[var(--ds-accent)] font-bold bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded">
                      Bytes &approx; Layers &times; Tokens &times; KV_Heads &times; Head_Dim &times; 2 (K, V) &times; Bytes_Per_Val
                    </div>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Interactive Calculator</span>
                  
                  <div className="space-y-3.5">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Query Heads (Q):</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{queryHeads}</span>
                      </div>
                      <input
                        type="range"
                        min="8"
                        max="64"
                        step="8"
                        value={queryHeads}
                        onChange={(e) => setQueryHeads(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Precision Bytes:</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{bytesPerValue === 2 ? 'FP16 (2B)' : bytesPerValue === 1 ? 'FP8 (1B)' : 'INT4 (0.5B)'}</span>
                      </div>
                      <div className="flex gap-1.5">
                        {[2, 1, 0.5].map((v) => (
                          <button
                            key={v}
                            data-math-control
                            onClick={() => setBytesPerValue(v)}
                            className={`ds-btn text-xs py-1 px-2.5 rounded font-mono ${
                              bytesPerValue === v
                                ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)]'
                                : 'bg-transparent text-[var(--ds-faint)] border border-[var(--ds-rule)] hover:bg-[var(--ds-paper)]'
                            }`}
                          >
                            {v} B
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-3 mt-4 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>MHA Cache Footprint:</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{calculatedMetrics.mhaGB} GB</span>
                    </div>
                    <div className="flex justify-between text-xs text-[var(--ds-accent)] font-bold">
                      <span>Active Selection Cache:</span>
                      <span className="font-mono text-sm">{calculatedMetrics.activeGB} GB</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 border-l-4 border-yellow-500 bg-yellow-50/50 p-4 text-xs text-yellow-950">
                <span className="font-bold block mb-1">Misconception Card</span>
                The KV cache is NOT the model weights. Weights are static and remain constant for a model size. The KV cache is dynamic per-request memory that grows linearly with prompt size and generated token count.
              </div>
            </div>
          )}

          {/* TAB 2: MHA LAYOUT */}
          {activeTab === 'mha-layout' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Layers className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Multi-Head Attention (MHA) Layout</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                  <p>
                    In standard Multi-Head Attention, each query head gets its own corresponding key and value projections. This maximizes representational expressiveness, allowing heads to track separate context details.
                  </p>
                  <p>
                    However, this is extremely memory-intensive during decoding. Since K/V is stored separately for all heads, context scaling maps to massive memory consumption.
                  </p>
                  <div className="border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-4 text-xs rounded">
                    <span className="font-bold text-[var(--ds-ink)] block mb-1">MHA Memory Formula:</span>
                    <span className="font-mono text-[var(--ds-accent)]">MHA Cache &propto; H_q &times; d_head</span>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block mb-3">MHA Memory Lanes</span>
                  
                  <div className="space-y-2">
                    {Array.from({ length: 4 }).map((_, hIdx) => (
                      <div key={hIdx} className="flex items-center gap-2.5 bg-[var(--ds-panel)] p-2 rounded border border-[var(--ds-rule)]">
                        <span className="text-[10px] font-bold font-mono text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">Query Head {hIdx+1}</span>
                        <span className="text-[var(--ds-mute)]">&rarr;</span>
                        <div className="flex gap-1">
                          <span className="text-[10px] font-mono bg-emerald-500 text-white px-2 py-0.5 rounded font-bold shadow-xs">Key {hIdx+1}</span>
                          <span className="text-[10px] font-mono bg-amber-500 text-white px-2 py-0.5 rounded font-bold shadow-xs">Value {hIdx+1}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[10px] text-[var(--ds-mute)] mt-4 leading-normal">
                    <strong>Takeaway:</strong> In MHA, each head has its own separate memory lane. No head shares cached information with another.
                  </p>
                </div>
              </div>

              <div className="mt-4 border-l-4 border-rose-500 bg-rose-50/50 p-4 text-xs text-rose-950">
                <span className="font-bold block mb-1">Misconception Card</span>
                Standard multi-head attention does not share a single context memory. Each attention head computes its own keys and values, which must be cached independently in every layer.
              </div>
            </div>
          )}

          {/* TAB 3: MQA/GQA SHARING */}
          {activeTab === 'mqa-gqa' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <GitBranch className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">KV Sharing: MQA and GQA Topologies</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                  <p>
                    To resolve MHA memory bloat, models adopt head-sharing designs:
                  </p>
                  <ul className="list-disc pl-4 space-y-1.5">
                    <li><strong>MQA (Multi-Query Attention)</strong>: All query heads share exactly one KV head. Extremely memory-efficient, but may degrade recall quality.</li>
                    <li><strong>GQA (Grouped-Query Attention)</strong>: Query heads are split into groups, and each group shares a single KV head pair (e.g. 4 query heads share 1 KV head). GQA recovers MHA quality while keeping speed close to MQA.</li>
                  </ul>
                  <div className="border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-4 text-xs rounded">
                    <span className="font-bold text-[var(--ds-ink)] block mb-1">Cache Reduction Ratio:</span>
                    <span className="font-mono text-[var(--ds-accent)]">Ratio = H_kv / H_q</span>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">KV Groups Configuration</span>
                  
                  <div className="space-y-3.5">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Query Heads (H_q):</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{queryHeads}</span>
                      </div>
                      <input
                        type="range"
                        min="8"
                        max="64"
                        step="8"
                        value={queryHeads}
                        onChange={(e) => setQueryHeads(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">KV Heads (H_kv):</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{kvHeads}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max={queryHeads}
                        step="1"
                        value={kvHeads}
                        onChange={(e) => setKvHeads(Math.min(Number(e.target.value), queryHeads))}
                        className="ds-range"
                      />
                    </div>
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-3 mt-4 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>Query Heads per Group:</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{(queryHeads / kvHeads).toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between text-[var(--ds-accent)] font-bold">
                      <span>Cache Size vs MHA:</span>
                      <span className="font-mono">{((kvHeads / queryHeads) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 border-l-4 border-yellow-500 bg-yellow-50/50 p-4 text-xs text-yellow-950">
                <span className="font-bold block mb-1">Misconception Card</span>
                GQA does not reduce the number of queries or query heads. It only reduces the number of key and value heads, which means multiple queries attend to the same key/value spaces.
              </div>
            </div>
          )}

          {/* TAB 4: MLA LATENT CACHE */}
          {activeTab === 'mla-latent' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Cpu className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">MLA: Compress K/V into Latent States</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3 text-sm text-[var(--ds-faint)] leading-relaxed">
                  <p>
                    <strong>Multi-head Latent Attention (MLA)</strong> takes a different path. Instead of sharing heads, MLA compresses the key and value representations into a narrow latent space using low-rank matrices.
                  </p>
                  <p>
                    During generation, we cache only the compressed latent vector <MathBlock>c_t</MathBlock>. Later, when computing attention, we up-project it. By absorbing up-projection directly into the query projection, we compute attention scores without reconstructing the key vectors in memory.
                  </p>
                  <div className="border border-[var(--ds-rule)] bg-[var(--ds-paper-2)] p-4 text-xs rounded space-y-1">
                    <span className="font-bold text-[var(--ds-ink)] block">Low-Rank Compression Formula:</span>
                    <div className="font-mono text-[var(--ds-accent)]">c_t = W_down &middot; x_t</div>
                    <div className="font-mono text-[var(--ds-accent)]">K_t, V_t &approx; W_up &middot; c_t</div>
                  </div>
                </div>

                <div className="bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">MLA Latent Settings</span>
                  
                  <div className="space-y-3.5">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Latent Dimension (d_c):</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{latentDim}</span>
                      </div>
                      <input
                        type="range"
                        min="32"
                        max="512"
                        step="32"
                        value={latentDim}
                        onChange={(e) => setLatentDim(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[var(--ds-faint)]">Head Dimension (d_h):</span>
                        <span className="font-mono font-bold text-[var(--ds-ink)]">{headDim}</span>
                      </div>
                      <select
                        value={headDim}
                        onChange={(e) => setHeadDim(Number(e.target.value))}
                        className="bg-transparent text-xs font-bold text-[var(--ds-ink)] outline-none border border-[var(--ds-rule)] p-1 rounded w-full"
                      >
                        <option value={64}>64</option>
                        <option value={128}>128</option>
                        <option value={192}>192</option>
                      </select>
                    </div>
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-3 mt-4 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>MLA Cache Footprint:</span>
                      <span className="font-mono font-bold text-[var(--ds-ink)]">{calculatedMetrics.mlaGB} GB</span>
                    </div>
                    <div className="flex justify-between text-[var(--ds-accent)] font-bold">
                      <span>Relative size vs MHA:</span>
                      <span className="font-mono">{calculatedMetrics.cacheRatioPercent}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 border-l-4 border-purple-500 bg-purple-50/50 p-4 text-xs text-purple-950">
                <span className="font-bold block mb-1">DeepSeek Paper Citation</span>
                "MLA guarantees efficient inference by significantly compressing KV cache into a latent vector. Specifically, MLA compresses KV cache to 128 dimensions, reducing KV cache by 93.3%." (DeepSeek-V2, arXiv)
              </div>
            </div>
          )}

          {/* TAB 5: MEMORY COMPARISON */}
          {activeTab === 'memory-comparison' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <BarChart2 className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">MHA vs GQA vs MLA Memory Scaling</h3>
              </div>

              <p className="text-xs text-[var(--ds-faint)]">
                Drag the context length slider to watch how each architecture's KV cache memory scales up to 1 million tokens. Notice that MHA quickly hits memory limit lines.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                
                {/* Sliders / controls */}
                <div className="md:col-span-4 space-y-4 bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded">
                  <div>
                    <label className="block text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-1">
                      Context Tokens: {contextLength >= 1000000 ? '1M' : (contextLength / 1000).toFixed(0) + 'k'}
                    </label>
                    <input
                      type="range"
                      min="1024"
                      max="256000"
                      step="4096"
                      value={contextLength}
                      onChange={(e) => setContextLength(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-1">
                      Memory Limit Budget: {memoryLimitGB} GB
                    </label>
                    <input
                      type="range"
                      min="8"
                      max="160"
                      step="8"
                      value={memoryLimitGB}
                      onChange={(e) => setMemoryLimitGB(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>
                </div>

                {/* Bars comparison (log scale-ish representing relative size) */}
                <div className="md:col-span-8 space-y-4">
                  
                  {/* MHA Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold">Multi-Head Attention (MHA)</span>
                      <span className="font-mono text-[var(--ds-faint)]">{calculatedMetrics.mhaGB} GB</span>
                    </div>
                    <div className="w-full bg-slate-200 h-6 rounded-xs overflow-hidden relative border border-[var(--ds-rule)]">
                      <span
                        className="bg-red-400 h-full block transition-all duration-300"
                        style={{ width: `${Math.min(100, (parseFloat(calculatedMetrics.mhaGB) / memoryLimitGB) * 100)}%` }}
                      />
                      {parseFloat(calculatedMetrics.mhaGB) > memoryLimitGB && (
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-red-950 uppercase bg-red-100/80">OUT OF MEMORY</span>
                      )}
                    </div>
                  </div>

                  {/* GQA Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold">Grouped-Query Attention (GQA)</span>
                      <span className="font-mono text-[var(--ds-faint)]">{calculatedMetrics.gqaGB} GB</span>
                    </div>
                    <div className="w-full bg-slate-200 h-6 rounded-xs overflow-hidden relative border border-[var(--ds-rule)]">
                      <span
                        className="bg-emerald-500 h-full block transition-all duration-300"
                        style={{ width: `${Math.min(100, (parseFloat(calculatedMetrics.gqaGB) / memoryLimitGB) * 100)}%` }}
                      />
                      {parseFloat(calculatedMetrics.gqaGB) > memoryLimitGB && (
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-red-950 uppercase bg-red-100/80">OUT OF MEMORY</span>
                      )}
                    </div>
                  </div>

                  {/* MLA Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold">Multi-Head Latent Attention (MLA)</span>
                      <span className="font-mono text-[var(--ds-faint)]">{calculatedMetrics.mlaGB} GB</span>
                    </div>
                    <div className="w-full bg-slate-200 h-6 rounded-xs overflow-hidden relative border border-[var(--ds-rule)]">
                      <span
                        className="bg-purple-500 h-full block transition-all duration-300"
                        style={{ width: `${Math.min(100, (parseFloat(calculatedMetrics.mlaGB) / memoryLimitGB) * 100)}%` }}
                      />
                      {parseFloat(calculatedMetrics.mlaGB) > memoryLimitGB && (
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-red-950 uppercase bg-red-100/80">OUT OF MEMORY</span>
                      )}
                    </div>
                  </div>

                </div>
              </div>
            </div>
          )}

          {/* TAB 6: DECODE BANDWIDTH */}
          {activeTab === 'decode-bandwidth' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Gauge className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Memory Bandwidth vs FLOPs</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                <div className="md:col-span-5 space-y-4 bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Hardware Profile</span>
                  
                  <div className="flex flex-col gap-1.5">
                    {[
                      { id: 'bandwidth-limited', name: 'RTX 4090 / Bandwidth-bound', desc: 'Slow memory bus, fast arithmetic.' },
                      { id: 'balanced', name: 'H100 SXM / Balanced', desc: 'SOTA High-Bandwidth Memory (HBM3).' },
                      { id: 'compute-limited', name: 'SRAM / Cache-Bound', desc: 'Extreme speed cache, compute limited.' }
                    ].map((hw) => (
                      <button
                        key={hw.id}
                        data-math-control
                        onClick={() => setHardwareProfile(hw.id)}
                        className={`ds-btn text-left p-2.5 rounded border text-xs ${
                          hardwareProfile === hw.id
                            ? 'bg-[var(--ds-accent-w)] border-[var(--ds-accent)] text-[var(--ds-accent)] font-bold'
                            : 'bg-transparent text-[var(--ds-faint)] border-[var(--ds-rule)] hover:bg-[var(--ds-paper)]'
                        }`}
                      >
                        <div className="font-bold">{hw.name}</div>
                        <div className="text-[10px] opacity-80 mt-0.5">{hw.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="md:col-span-7 space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Bandwidth Pipe Saturation</span>
                  
                  <div className="border border-[var(--ds-rule)] p-4 rounded bg-slate-100/50 flex flex-col justify-center items-center min-h-[140px] relative">
                    
                    {/* Pipe simulation */}
                    <div className="w-full bg-slate-200 h-10 rounded border border-[var(--ds-rule)] overflow-hidden relative flex items-center">
                      <span
                        className="h-full block transition-all duration-500"
                        style={{
                          width: `${calculatedMetrics.busSaturation}%`,
                          backgroundColor: parseFloat(calculatedMetrics.busSaturation) > 85 ? 'var(--ds-warm)' : 'var(--ds-accent)'
                        }}
                      />
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-mono font-bold text-[var(--ds-ink)]">
                        Memory Pipe Saturation: {calculatedMetrics.busSaturation}%
                      </span>
                    </div>

                    <p className="text-[11px] text-[var(--ds-faint)] text-center mt-3">
                      At context = {contextLength} tokens, generating a single token requires moving{' '}
                      <span className="font-bold text-[var(--ds-ink)]">{(parseFloat(calculatedMetrics.readPerStepMB) * concurrentRequests).toFixed(1)} MB</span>{' '}
                      of cached vectors from GPU memory to processor registers.
                    </p>
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-3 flex justify-between text-xs font-mono">
                    <div>
                      <span className="block text-[10px] text-[var(--ds-faint)]">PROJECTION WORK</span>
                      <span className="text-[var(--ds-accent)] font-bold">{calculatedMetrics.flopsMflops} MFLOPs</span>
                    </div>
                    <div className="text-right">
                      <span className="block text-[10px] text-[var(--ds-faint)]">DECODE STATUS</span>
                      <span className={parseFloat(calculatedMetrics.busSaturation) > 80 ? 'text-[var(--ds-warm)] font-bold' : 'text-emerald-600 font-bold'}>
                        {parseFloat(calculatedMetrics.busSaturation) > 80 ? 'Memory Bandwidth Saturation!' : 'Execution Balanced'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 7: QUALITY VS MEMORY TRADEOFF */}
          {activeTab === 'quality-memory' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <TrendingUp className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Quality vs Memory Tradeoff Frontier</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                <div className="md:col-span-5 space-y-4 bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Constraint Budget</span>
                  
                  <div className="space-y-3.5">
                    <div>
                      <label className="block text-xs font-mono text-[var(--ds-faint)] mb-1">Max Cache Memory: {memoryBudget} GB</label>
                      <input
                        type="range"
                        min="2"
                        max="80"
                        step="2"
                        value={memoryBudget}
                        onChange={(e) => setMemoryBudget(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-mono text-[var(--ds-faint)] mb-1">Target Context: {qualityTargetContext / 1000}k tokens</label>
                      <input
                        type="range"
                        min="4096"
                        max="128000"
                        step="4096"
                        value={qualityTargetContext}
                        onChange={(e) => setQualityTargetContext(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                  </div>
                </div>

                <div className="md:col-span-7 space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Architectural Tradeoff Ledger</span>
                  
                  <div className="space-y-2">
                    {[
                      { id: 'mha', name: 'MHA Baseline', size: calculatedMetrics.mhaGB, risk: 5, desc: 'Highest recall quality, massive cache footprint.' },
                      { id: 'gqa', name: 'GQA Grouped (8 heads)', size: calculatedMetrics.gqaGB, risk: 25, desc: 'Strong intermediate choice, medium footprint.' },
                      { id: 'mla', name: 'MLA Compressed (128d)', size: calculatedMetrics.mlaGB, risk: 15, desc: 'SOTA compression ratio, low risk, extra projections.' }
                    ].map((item) => {
                      const fit = parseFloat(item.size) <= memoryBudget;
                      return (
                        <div
                          key={item.id}
                          className={`p-3 rounded border text-xs transition-all ${
                            fit
                              ? 'border-emerald-200 bg-emerald-50/40 text-emerald-950'
                              : 'border-rose-200 bg-rose-50/40 text-rose-950 opacity-60'
                          }`}
                        >
                          <div className="flex justify-between items-center font-bold">
                            <span>{item.name}</span>
                            <span className="font-mono">
                              {item.size} GB ({fit ? 'Fits Budget' : 'Exceeds Budget'})
                            </span>
                          </div>
                          <p className="opacity-80 mt-1">{item.desc}</p>
                          <div className="mt-2 flex justify-between text-[10px]">
                            <span>Quality Risk Indicator:</span>
                            <span className="font-bold">{item.risk}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 8: LONG-CONTEXT PLAYGROUND */}
          {activeTab === 'serving-playground' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <Server className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Long-Context Multi-User Serving Playground</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                
                {/* Control Panel */}
                <div className="md:col-span-4 space-y-4 bg-[var(--ds-paper-2)] p-4 border border-[var(--ds-rule)] rounded">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Serving Parameters</span>
                  
                  <div>
                    <label className="block text-xs font-mono text-[var(--ds-faint)] mb-1">Concurrent Queries: {concurrentRequests}</label>
                    <input
                      type="range"
                      min="1"
                      max="64"
                      step="1"
                      value={concurrentRequests}
                      onChange={(e) => setConcurrentRequests(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-mono text-[var(--ds-faint)] mb-1">GPU Serving Memory: {memoryLimitGB} GB</label>
                    <input
                      type="range"
                      min="24"
                      max="160"
                      step="8"
                      value={memoryLimitGB}
                      onChange={(e) => setMemoryLimitGB(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>
                </div>

                {/* Queue Allocator Graphics */}
                <div className="md:col-span-8 space-y-4">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">GPU Memory Allocator</span>
                  
                  <div className="border border-[var(--ds-rule)] p-4 rounded bg-slate-100/50 min-h-[160px]">
                    <div className="flex justify-between items-center text-xs mb-3 font-mono">
                      <span>Total Serving Memory</span>
                      <span className={calculatedMetrics.fitsBudget ? 'text-emerald-600 font-bold' : 'text-rose-600 font-bold'}>
                        {calculatedMetrics.totalRequestGb} / {memoryLimitGB} GB
                      </span>
                    </div>

                    {/* Allocation grid */}
                    <div className="flex flex-wrap gap-1 border border-[var(--ds-rule)] p-2 bg-[var(--ds-panel)] min-h-[80px] rounded">
                      {Array.from({ length: concurrentRequests }).map((_, rIdx) => {
                        const costPerReq = parseFloat(calculatedMetrics.activeGB);
                        const fit = (rIdx + 1) * costPerReq <= memoryLimitGB;
                        return (
                          <div
                            key={rIdx}
                            className={`w-8 h-8 rounded-xs flex flex-col items-center justify-center border text-[9px] font-bold shadow-xs ${
                              fit
                                ? 'bg-emerald-100 border-emerald-300 text-emerald-950'
                                : 'bg-red-100 border-red-300 text-red-950 animate-pulse'
                            }`}
                          >
                            <span>U{rIdx+1}</span>
                            <span className="text-[8px] opacity-75 font-mono">{(costPerReq).toFixed(1)}G</span>
                          </div>
                        );
                      })}
                    </div>

                    <p className="text-[11px] text-[var(--ds-faint)] mt-3 leading-normal">
                      Each query consumes <span className="font-bold text-[var(--ds-ink)]">{calculatedMetrics.activeGB} GB</span> of KV cache. 
                      {calculatedMetrics.fitsBudget 
                        ? ' All concurrent request caches successfully fit inside GPU serving budget.'
                        : ' Memory limit exceeded! Serving will trigger out-of-memory crashes or require paging.'
                      }
                    </p>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* TAB 9: PAPER DECODER */}
          {activeTab === 'paper-decoder' && (
            <div className="space-y-4 animate-fade-in">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-2 mb-3">
                <BookOpen className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-md font-bold text-[var(--ds-ink)]">Paper Decoder: Modern Citations Analyzer</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                
                {/* Citations index */}
                <div className="md:col-span-5 space-y-2 bg-[var(--ds-paper-2)] p-3 border border-[var(--ds-rule)] rounded">
                  <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block mb-2">Paper Anchors</span>
                  
                  {PAPER_ANCHORS.map((p) => (
                    <button
                      key={p.id}
                      data-math-control
                      onClick={() => setPaperSelectedText(p.id)}
                      className={`ds-btn w-full text-left p-2 rounded text-xs transition-all ${
                        paperSelectedText === p.id
                          ? 'bg-[var(--ds-accent)] text-[var(--ds-paper)] font-bold'
                          : 'bg-transparent text-[var(--ds-faint)] border border-[var(--ds-rule)] hover:bg-[var(--ds-paper)]'
                      }`}
                    >
                      {p.title}
                    </button>
                  ))}
                </div>

                {/* Selected citation text */}
                <div className="md:col-span-7 space-y-4">
                  {(() => {
                    const p = PAPER_ANCHORS.find((item) => item.id === paperSelectedText);
                    if (!p) return null;
                    return (
                      <div className="border border-[var(--ds-rule)] p-4 rounded bg-slate-100/50 space-y-3">
                        <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Citation Excerpt</span>
                        <blockquote className="text-xs text-[var(--ds-ink)] italic border-l-4 border-[var(--ds-accent)] pl-3 py-1 bg-[var(--ds-panel)] border-[var(--ds-rule)] rounded-r">
                          "{p.takeaway}"
                        </blockquote>
                        <div className="text-[10px] font-mono text-[var(--ds-mute)] text-right">
                          {p.citation}
                        </div>
                      </div>
                    );
                  })()}

                  {/* Self Check Question */}
                  <div className="border border-[var(--ds-rule)] p-4 rounded bg-[var(--ds-paper-2)] space-y-3">
                    <span className="text-[10px] font-bold text-[var(--ds-faint)] uppercase tracking-wider block">Paper Self-Check Question</span>
                    <p className="text-xs text-[var(--ds-ink)]">
                      "Our model uses 32 query heads, 4 KV heads, and a compressed latent KV dimension of 128..."
                    </p>
                    <div className="space-y-1.5">
                      {[
                        { key: 'a', text: 'This is GQA layout since KV heads is 4.' },
                        { key: 'b', text: 'This is MLA layout caching a latent KV state.' },
                        { key: 'c', text: 'This is standard MHA since it has 32 query heads.' }
                      ].map((ans) => (
                        <button
                          key={ans.key}
                          data-math-control
                          onClick={() => setPaperAnswers({ ...paperAnswers, q1: ans.key })}
                          className={`ds-btn w-full text-left p-2 rounded text-xs border transition-all ${
                            paperAnswers.q1 === ans.key
                              ? 'bg-[var(--ds-accent-w)] border-[var(--ds-accent)] text-[var(--ds-accent)] font-bold'
                              : 'bg-[var(--ds-panel)] border-[var(--ds-rule)] text-[var(--ds-faint)] hover:border-[var(--ds-ink)]'
                          }`}
                        >
                          {ans.text}
                        </button>
                      ))}
                    </div>

                    {paperAnswers.q1 && (
                      <div className={`p-2.5 rounded text-xs ${
                        paperAnswers.q1 === 'b'
                          ? 'bg-emerald-50 text-emerald-950 border border-emerald-200'
                          : 'bg-rose-50 text-rose-950 border border-rose-200'
                      }`}>
                        {paperAnswers.q1 === 'b'
                          ? 'Correct! Because it mentions caching a compressed latent KV dimension of 128, which is the defining signature of MLA.'
                          : 'Incorrect. While it has 4 KV heads, the presence of a compressed latent KV dimension of 128 points to MLA cache representation.'
                        }
                      </div>
                    )}
                  </div>
                </div>

              </div>
            </div>
          )}

        </div>

      </div>
      
    </div>
  );
}
