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
  Zap,
  Gauge,
  TrendingUp,
  Sparkles,
  RefreshCw,
  Flame,
  FileText,
  Filter,
  CheckCircle,
  XCircle,
  HelpCircle,
  Maximize2
} from 'lucide-react';
import {
  REASONING_PIPELINE_STAGES,
  SFT_TRACES_EXAMPLES,
  REJECTION_SAMPLING_SAMPLES,
  ORM_VS_PRM_SAMPLES,
  GRPO_WORKBENCH_PROMPTS,
  COLD_START_VS_PURE_RL,
  FAILURE_MODES,
  DISTILLATION_PIPELINES,
  PAPER_DECODERS
} from './data';

const TABS = [
  { id: 'training-map', label: 'Training Map', icon: Network },
  { id: 'sft-traces', label: 'SFT Traces', icon: FileText },
  { id: 'rejection-sampling', label: 'Rejection Sampling', icon: Filter },
  { id: 'orm-vs-prm', label: 'ORM vs PRM', icon: Layers },
  { id: 'rlvr-rewards', label: 'RLVR Rewards', icon: Gauge },
  { id: 'grpo-workbench', label: 'GRPO Workbench', icon: Cpu },
  { id: 'cold-start-vs-rl', label: 'Cold Start vs Pure RL', icon: Flame },
  { id: 'failure-modes', label: 'Failure Modes', icon: AlertTriangle },
  { id: 'distillation', label: 'Distillation', icon: Sparkles },
  { id: 'paper-decoder', label: 'Paper Decoder', icon: Brain }
];

export default function ReasoningRlvrGrpo() {
  const [activeTab, setActiveTab] = useState('training-map');

  // Training Map State
  const [selectedStage, setSelectedStage] = useState('rlvr-grpo');

  // SFT Traces State
  const [selectedTraceId, setSelectedTraceId] = useState('correct-structured');
  const activeTraceObj = useMemo(() => {
    return SFT_TRACES_EXAMPLES.find(t => t.id === selectedTraceId) || SFT_TRACES_EXAMPLES[0];
  }, [selectedTraceId]);

  // Rejection Sampling State
  const [strictness, setStrictness] = useState(0.85); // 0.0 to 1.0
  const [rsCandidates, setRsCandidates] = useState(REJECTION_SAMPLING_SAMPLES);
  const [isGeneratingRS, setIsGeneratingRS] = useState(false);

  const filteredRSCandidates = useMemo(() => {
    return rsCandidates.map(c => {
      // Correctness determines base quality. Strictness scales how hard it is to pass.
      const rawQuality = c.status === 'Correct' ? 0.95 : (c.status.includes('Format Fail') ? 0.8 : 0.2);
      // Format checks
      const isFormatCorrect = !c.status.includes('Format') && !c.status.includes('No Box') && !c.status.includes('Empty');
      const passesFilter = rawQuality >= strictness && (strictness < 0.5 || isFormatCorrect);
      return { ...c, passesFilter };
    });
  }, [rsCandidates, strictness]);

  const rsStats = useMemo(() => {
    const passed = filteredRSCandidates.filter(c => c.passesFilter).length;
    const total = filteredRSCandidates.length;
    const rate = total > 0 ? Math.round((passed / total) * 100) : 0;
    return { passed, total, rate };
  }, [filteredRSCandidates]);

  // ORM vs PRM State
  const [useAlternativeORM, setUseAlternativeORM] = useState(false);
  const activeORMSteps = useMemo(() => {
    return useAlternativeORM ? ORM_VS_PRM_SAMPLES.alternativeWrongFinal : ORM_VS_PRM_SAMPLES.steps;
  }, [useAlternativeORM]);

  // RLVR Reward Tuner State
  const [correctWeight, setCorrectWeight] = useState(1.0);
  const [formatWeight, setFormatWeight] = useState(0.2);
  const [langWeight, setLangWeight] = useState(0.2);
  const [lengthPenalty, setLengthPenalty] = useState(0.001); // penalty per token

  const liveRewardsData = useMemo(() => {
    return SFT_TRACES_EXAMPLES.map(trace => {
      // Calculate active scores based on current sliders
      const correctScore = trace.stats.correctness * correctWeight;
      const formatScore = (trace.id === 'reward-hacking-format' ? 2.5 : trace.stats.format) * formatWeight;
      const langScore = trace.stats.language * langWeight;
      const lengthScore = -(trace.stats.length * lengthPenalty);
      
      const rawSum = correctScore + formatScore + langScore + lengthScore;
      const totalReward = parseFloat(rawSum.toFixed(3));
      
      return {
        ...trace,
        liveScores: { correctScore, formatScore, langScore, lengthScore, totalReward }
      };
    });
  }, [correctWeight, formatWeight, langWeight, lengthPenalty]);

  // GRPO Workbench State
  const [grpoPromptIndex, setGrpoPromptIndex] = useState(0);
  const grpoPromptObj = GRPO_WORKBENCH_PROMPTS[grpoPromptIndex];
  
  // Weights for GRPO simulator
  const [grpoCorrectWeight, setGrpoCorrectWeight] = useState(1.0);
  const [grpoFormatWeight, setGrpoFormatWeight] = useState(0.3);
  const [grpoLangWeight, setGrpoLangWeight] = useState(0.2);
  const [grpoLengthPenalty, setGrpoLengthPenalty] = useState(0.0005);
  const [grpoClipEpsilon, setGrpoClipEpsilon] = useState(0.2);
  const [grpoGroupSize, setGrpoGroupSize] = useState(8);

  const [grpoHistory, setGrpoHistory] = useState([78, 80, 81, 82, 85, 87]);
  const [stepCount, setStepCount] = useState(0);

  // Compute live GRPO candidate metrics
  const grpoCalculations = useMemo(() => {
    const rawCandidates = grpoPromptObj.candidates.slice(0, grpoGroupSize);
    
    // 1. Calculate raw rewards for each candidate in group
    const scoredCandidates = rawCandidates.map(c => {
      const isCorrect = c.rewards.correct;
      const isFormat = c.rewards.format;
      const isLang = c.rewards.lang;
      const length = Math.abs(c.rewards.length * 1000); // map back to token count approx
      
      const rCorrect = isCorrect * grpoCorrectWeight;
      const rFormat = isFormat * grpoFormatWeight;
      const rLang = isLang * grpoLangWeight;
      const rLength = -(length * grpoLengthPenalty);
      const totalReward = parseFloat((rCorrect + rFormat + rLang + rLength).toFixed(4));

      return {
        ...c,
        metrics: { rCorrect, rFormat, rLang, rLength, totalReward, length }
      };
    });

    // 2. Compute group mean and std
    const rewards = scoredCandidates.map(c => c.metrics.totalReward);
    const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
    
    const variance = rewards.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / rewards.length;
    const std = Math.sqrt(variance);
    const epsilon = 1e-6; // stabilizer

    // 3. Compute group relative advantages
    const finalCandidates = scoredCandidates.map(c => {
      const advantage = std > 0.001 
        ? (c.metrics.totalReward - mean) / (std + epsilon)
        : 0.0;
      
      // Calculate policy gradient update direction
      // Positive advantage increases selection probability, clipped by clip epsilon
      const updateStrength = Math.min(Math.max(advantage, -2.0), 2.0);
      const isClipped = Math.abs(updateStrength) > grpoClipEpsilon;
      const clipUpdate = isClipped 
        ? (updateStrength > 0 ? grpoClipEpsilon : -grpoClipEpsilon)
        : updateStrength;

      return {
        ...c,
        metrics: {
          ...c.metrics,
          advantage: parseFloat(advantage.toFixed(4)),
          clipUpdate: parseFloat(clipUpdate.toFixed(4)),
          isClipped
        }
      };
    });

    // Check if group is all-negative
    const isAllNegative = rewards.every(r => r <= 0.01);

    return {
      candidates: finalCandidates,
      mean: parseFloat(mean.toFixed(4)),
      std: parseFloat(std.toFixed(4)),
      isAllNegative
    };
  }, [grpoPromptObj, grpoCorrectWeight, grpoFormatWeight, grpoLangWeight, grpoLengthPenalty, grpoGroupSize, grpoClipEpsilon]);

  // Failure Modes State
  const [activeFailureId, setActiveFailureId] = useState('overthinking');
  const [appliedFixes, setAppliedFixes] = useState({
    overthinking: false,
    'reward-hacking': false,
    'language-mixing': false,
    'kl-collapse': false
  });

  const currentFailureObj = useMemo(() => {
    return FAILURE_MODES.find(f => f.id === activeFailureId) || FAILURE_MODES[0];
  }, [activeFailureId]);

  // Paper Decoder Quiz State
  const [selectedPaperQuoteIndex, setSelectedPaperQuoteIndex] = useState(0);
  const [selectedPaperAnswer, setSelectedPaperAnswer] = useState(null);
  const [showPaperExplanation, setShowPaperExplanation] = useState(false);

  const activeQuote = useMemo(() => {
    const r1 = PAPER_DECODERS[0];
    return r1.quotes[selectedPaperQuoteIndex];
  }, [selectedPaperQuoteIndex]);

  // Interactive functions
  const handleGenerateRS = () => {
    setIsGeneratingRS(true);
    setTimeout(() => {
      const randomized = REJECTION_SAMPLING_SAMPLES.map(item => {
        const coin = Math.random();
        if (coin > 0.6) {
          return {
            ...item,
            id: 'gen-' + Math.random().toString(36).substr(2, 4),
            trace: `<think> Backtracking check... verified. </think> \\boxed{${Math.random() > 0.3 ? '21' : '15'}}`,
            status: Math.random() > 0.3 ? 'Correct' : 'Wrong',
            length: Math.floor(Math.random() * 80) + 40
          };
        }
        return item;
      });
      setRsCandidates(randomized);
      setIsGeneratingRS(false);
    }, 800);
  };

  const handleStepGRPO = () => {
    // Simulate policy updating towards higher accuracy
    setStepCount(prev => prev + 1);
    
    // Policy accuracy gains are stronger if reward weights are balanced
    const hasLengthPenalty = grpoLengthPenalty > 0.0001;
    const isCorrectWeightDom = grpoCorrectWeight > grpoFormatWeight * 1.5;
    const isClipReasonable = grpoClipEpsilon >= 0.1 && grpoClipEpsilon <= 0.3;
    
    let accuracyGain = 0;
    if (isCorrectWeightDom && hasLengthPenalty && isClipReasonable) {
      accuracyGain = Math.floor(Math.random() * 4) + 1; // standard positive step
    } else if (grpoCalculations.isAllNegative) {
      accuracyGain = -Math.floor(Math.random() * 3) - 1; // penalty step
    } else {
      accuracyGain = Math.random() > 0.5 ? 1 : -1; // random noisy step
    }

    setGrpoHistory(prev => {
      const nextAcc = Math.min(Math.max(prev[prev.length - 1] + accuracyGain, 50), 99);
      return [...prev.slice(1), nextAcc];
    });
  };

  return (
    <div className="flex flex-col min-h-screen bg-[var(--ds-paper)] text-[var(--ds-ink)] font-sans antialiased">
      {/* Header and metadata details */}
      <div className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase bg-[var(--ds-warm)] border border-[var(--ds-rule)] text-[var(--ds-ink)] rounded">
                Frontier LLMs
              </span>
              <span className="px-2 py-0.5 text-[10px] font-bold tracking-wider uppercase bg-amber-500 text-white rounded">
                Advanced Module
              </span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-[var(--ds-ink)] font-display">
              Reasoning Models: From SFT to RLVR / GRPO
            </h1>
            <p className="text-xs text-[var(--ds-faint)] mt-1 max-w-2xl">
              Understand the mechanics of post-training scaling, verifiable reinforcement learning feedback loop systems, 
              group advantages, step-level credit mapping, formatting hacking exploits, and model distillation.
            </p>
          </div>
          <div className="flex items-center gap-4 bg-[var(--ds-paper)] p-3 border border-[var(--ds-rule)] rounded">
            <div className="text-center border-r border-[var(--ds-rule)] pr-4">
              <span className="block text-[10px] font-bold text-[var(--ds-faint)] uppercase">Active Compute</span>
              <span className="text-lg font-bold text-[var(--ds-accent)]">Test-Time Search</span>
            </div>
            <div className="text-center pl-2">
              <span className="block text-[10px] font-bold text-[var(--ds-faint)] uppercase">SOTA Algorithm</span>
              <span className="text-lg font-bold text-[var(--ds-ink)]">GRPO Policy</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs Navigation Grid */}
      <div className="border-b border-[var(--ds-rule)] bg-[var(--ds-panel)] sticky top-0 z-30 overflow-x-auto">
        <div className="max-w-7xl mx-auto flex">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                data-math-control
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-4 text-xs font-bold whitespace-nowrap transition-all border-b-2 ${
                  isActive
                    ? 'border-[var(--ds-accent)] text-[var(--ds-accent)] bg-[var(--ds-paper)]'
                    : 'border-transparent text-[var(--ds-faint)] hover:text-[var(--ds-ink)] hover:bg-[var(--ds-paper-2)]'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content wrapper */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 gap-6">
        
        {/* Tab content panels */}
        {activeTab === 'training-map' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded">
              <div className="flex items-center gap-2 mb-4">
                <Network className="w-5 h-5 text-[var(--ds-accent)]" />
                <h2 className="text-md font-bold uppercase tracking-wider">Post-Training Reasoning Pipeline Map</h2>
              </div>
              <p className="text-xs text-[var(--ds-faint)] mb-6">
                Click on any step below to trace how standard base language models are transformed into thinking-capable reasoning models.
              </p>

              {/* Graphic Flow Layout */}
              <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">
                {REASONING_PIPELINE_STAGES.map(stage => {
                  const isSelected = selectedStage === stage.id;
                  return (
                    <button
                      key={stage.id}
                      data-math-control
                      onClick={() => setSelectedStage(stage.id)}
                      className={`p-4 border text-left rounded transition-all flex flex-col justify-between h-36 ${
                        isSelected
                          ? 'border-[var(--ds-accent)] bg-[var(--ds-warm)] shadow-sm'
                          : 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]'
                      }`}
                    >
                      <div>
                        <span className="text-[10px] block font-bold text-[var(--ds-faint)] uppercase">Stage</span>
                        <h4 className="text-xs font-bold text-[var(--ds-ink)] mt-1">{stage.name}</h4>
                      </div>
                      <span className="text-[10px] text-[var(--ds-faint)] mt-auto block font-mono">
                        {stage.metrics.alignment === 'Emergent' ? 'RL Loop' : `Align: ${stage.metrics.alignment}`}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Detail Card panel */}
              {(() => {
                const stageObj = REASONING_PIPELINE_STAGES.find(s => s.id === selectedStage);
                return (
                  <div className="p-5 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="md:col-span-2 space-y-3">
                      <h3 className="text-sm font-bold text-[var(--ds-ink)]">{stageObj.name} Details</h3>
                      <p className="text-xs text-[var(--ds-faint)] leading-relaxed">{stageObj.description}</p>
                      
                      <div className="pt-2">
                        <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Key Mechanisms:</span>
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {stageObj.features.map((feat, i) => (
                            <li key={i} className="text-xs flex items-center gap-2 text-[var(--ds-ink)]">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--ds-accent)]" />
                              {feat}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="bg-[var(--ds-panel)] p-4 border border-[var(--ds-rule)] rounded space-y-3">
                      <h4 className="text-[10px] font-bold tracking-wider uppercase text-[var(--ds-faint)]">Stage KPI Scoreboard</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs border-b border-[var(--ds-rule)] pb-1.5">
                          <span className="text-[var(--ds-faint)]">Model Capacity:</span>
                          <span className="font-bold font-mono">{stageObj.metrics.capacity}</span>
                        </div>
                        <div className="flex justify-between text-xs border-b border-[var(--ds-rule)] pb-1.5">
                          <span className="text-[var(--ds-faint)]">Alignment Level:</span>
                          <span className="font-bold font-mono">{stageObj.metrics.alignment}</span>
                        </div>
                        <div className="flex justify-between text-xs pb-1">
                          <span className="text-[var(--ds-faint)]">Inference Latency:</span>
                          <span className="font-bold text-[var(--ds-accent)]">{stageObj.metrics.latency}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {activeTab === 'sft-traces' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Sidebar trace selector */}
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 rounded space-y-3">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-3 mb-2">
                <FileText className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--ds-ink)]">SFT Trace Curation</h3>
              </div>
              <div className="space-y-2">
                {SFT_TRACES_EXAMPLES.map(trace => {
                  const isSelected = selectedTraceId === trace.id;
                  return (
                    <button
                      key={trace.id}
                      data-math-control
                      onClick={() => setSelectedTraceId(trace.id)}
                      className={`w-full text-left p-3 rounded border text-xs transition-all ${
                        isSelected
                          ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] font-bold'
                          : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] hover:bg-[var(--ds-paper)]'
                      }`}
                    >
                      {trace.title}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Main trace viewer */}
            <div className="md:col-span-2 border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 rounded space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--ds-rule)] pb-3">
                <h3 className="text-sm font-bold text-[var(--ds-ink)]">{activeTraceObj.title} Curation Audit</h3>
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                  activeTraceObj.stats.correctness === 1.0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {activeTraceObj.stats.correctness === 1.0 ? 'PASSED CHECK' : 'FAILED CHECK'}
                </span>
              </div>

              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Prompt:</span>
                <p className="text-xs font-mono bg-[var(--ds-paper)] p-2.5 rounded border border-[var(--ds-rule)] text-[var(--ds-ink)]">
                  {activeTraceObj.prompt}
                </p>
              </div>

              <div>
                <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Generated Output Trace:</span>
                <pre className="text-[11px] font-mono bg-stone-900 text-stone-200 p-4 rounded overflow-x-auto max-h-72 leading-relaxed">
                  {activeTraceObj.text}
                </pre>
              </div>

              <div className="p-4 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Auditor Evaluation:</span>
                  <p className="text-xs text-[var(--ds-ink)] leading-relaxed">{activeTraceObj.evaluation}</p>
                </div>
                <div className="space-y-2 border-t md:border-t-0 md:border-l border-[var(--ds-rule)] pt-2 md:pt-0 md:pl-4 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-[var(--ds-faint)]">Estimated Length:</span>
                    <span>{activeTraceObj.stats.length} tokens</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--ds-faint)]">Format Score:</span>
                    <span>{activeTraceObj.stats.format}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[var(--ds-faint)]">Linguistic Coherence:</span>
                    <span>{activeTraceObj.stats.language}</span>
                  </div>
                  <div className="flex justify-between font-bold text-[var(--ds-accent)] pt-1 border-t border-[var(--ds-rule)]">
                    <span>Baseline Reward:</span>
                    <span>{activeTraceObj.stats.totalReward}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'rejection-sampling' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center justify-between flex-wrap gap-4 border-b border-[var(--ds-rule)] pb-4">
                <div className="flex items-center gap-2">
                  <Filter className="w-5 h-5 text-[var(--ds-accent)]" />
                  <h2 className="text-md font-bold uppercase tracking-wider">Rejection Sampling Dataset Curation</h2>
                </div>
                
                <button
                  data-math-control
                  disabled={isGeneratingRS}
                  onClick={handleGenerateRS}
                  className="ds-btn bg-[var(--ds-accent)] hover:bg-stone-800 text-[var(--ds-paper)] text-xs font-bold py-1.5 px-4 rounded transition-all inline-flex items-center gap-2 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isGeneratingRS ? 'animate-spin' : ''}`} />
                  {isGeneratingRS ? 'GENERATING SAMPLES...' : 'GENERATE SYNTHETIC CANDIDATES'}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Controller Panel */}
                <div className="space-y-5 bg-[var(--ds-paper)] p-4 border border-[var(--ds-rule)] rounded">
                  <h3 className="text-xs font-bold uppercase text-[var(--ds-ink)] tracking-wider">Verifier strictness filter</h3>
                  
                  <div>
                    <label className="block text-[11px] font-bold text-[var(--ds-faint)] uppercase tracking-wider mb-2">
                      Strictness Threshold: {Math.round(strictness * 100)}%
                    </label>
                    <input
                      type="range"
                      min="0.10"
                      max="1.00"
                      step="0.05"
                      value={strictness}
                      onChange={(e) => setStrictness(Number(e.target.value))}
                      className="ds-range"
                    />
                    <span className="text-[10px] text-[var(--ds-faint)] block mt-1 leading-relaxed">
                      Higher strictness demands correct final answers and exact tag formats. Lower strictness allows verbose, noisy, or half-formatted correct solutions.
                    </span>
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-4 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-[var(--ds-faint)]">Total Samples Audited:</span>
                      <span className="font-bold">{rsStats.total}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--ds-faint)]">Passed Filter:</span>
                      <span className="font-bold text-green-700">{rsStats.passed}</span>
                    </div>
                    <div className="flex justify-between border-t border-[var(--ds-rule)] pt-2 font-bold text-sm">
                      <span className="text-[var(--ds-faint)]">Yield Rate:</span>
                      <span className={rsStats.rate > 40 ? 'text-green-700' : 'text-red-700'}>{rsStats.rate}%</span>
                    </div>
                  </div>
                </div>

                {/* Candidate list view */}
                <div className="md:col-span-2 space-y-3">
                  <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Generated Trace Candidates Status:</span>
                  <div className="max-h-72 overflow-y-auto space-y-2 pr-2">
                    {filteredRSCandidates.map(cand => (
                      <div
                        key={cand.id}
                        className={`p-3 border rounded text-xs grid grid-cols-3 items-center transition-all ${
                          cand.passesFilter 
                            ? 'border-green-300 bg-green-50/50' 
                            : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] opacity-80'
                        }`}
                      >
                        <div className="col-span-2 pr-3 font-mono truncate">
                          {cand.trace}
                        </div>
                        <div className="flex items-center justify-end gap-2">
                          <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                            cand.status.includes('Correct') ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                          }`}>
                            {cand.status}
                          </span>
                          {cand.passesFilter ? (
                            <CheckCircle className="w-4 h-4 text-green-600 shrink-0" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-600 shrink-0" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'orm-vs-prm' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center justify-between border-b border-[var(--ds-rule)] pb-4 flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-[var(--ds-accent)]" />
                  <h2 className="text-md font-bold uppercase tracking-wider">Outcome vs Process Reward Modeling</h2>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[var(--ds-faint)] uppercase">Scenario Picker:</span>
                  <button
                    data-math-control
                    onClick={() => setUseAlternativeORM(!useAlternativeORM)}
                    className="ds-btn bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)] border border-[var(--ds-rule)] text-xs font-bold py-1 px-3 rounded transition-all"
                  >
                    {useAlternativeORM ? 'Switch to Standard Arithmetic Error' : 'Switch to Double-Negative Hack'}
                  </button>
                </div>
              </div>

              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                Outcome Reward Models (ORM) evaluate only the final answer (Step 6), while Process Reward Models (PRM) evaluate 
                each step of the reasoning path. PRMs prevent the policy from getting rewarded for logic errors or format hacking tricks.
              </p>

              <div className="bg-[var(--ds-paper)] p-4 border border-[var(--ds-rule)] rounded">
                <div className="text-xs font-mono font-bold mb-2">Prompt: {ORM_VS_PRM_SAMPLES.prompt}</div>
                <div className="space-y-2">
                  {activeORMSteps.map((step, idx) => {
                    let textClass = 'text-[var(--ds-ink)]';
                    let borderClass = 'border-[var(--ds-rule)]';
                    if (step.status === 'error') {
                      textClass = 'text-red-800';
                      borderClass = 'border-red-300 bg-red-50/50';
                    } else if (step.status === 'final-hack') {
                      textClass = 'text-amber-800 font-bold';
                      borderClass = 'border-amber-300 bg-amber-50/50';
                    } else if (step.status === 'correct-follow') {
                      textClass = 'text-blue-800';
                      borderClass = 'border-blue-200 bg-blue-50/20';
                    }

                    return (
                      <div
                        key={idx}
                        className={`p-3 border rounded text-xs grid grid-cols-1 md:grid-cols-4 items-center gap-3 transition-all ${borderClass} ${textClass}`}
                      >
                        <div className="md:col-span-2 font-mono leading-relaxed">
                          {step.text}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-center text-[10px] font-bold font-mono">
                          <div>
                            <span className="block text-[8px] text-[var(--ds-faint)] uppercase">ORM Score</span>
                            <span>{step.ormScore}</span>
                          </div>
                          <div>
                            <span className="block text-[8px] text-[var(--ds-faint)] uppercase">PRM Score</span>
                            <span className={step.prmScore.includes('+') ? 'text-green-700' : (step.prmScore.includes('-') ? 'text-red-700' : '')}>
                              {step.prmScore}
                            </span>
                          </div>
                        </div>
                        <div className="text-[10px] text-[var(--ds-faint)]">
                          {step.explanation || 'Step checks out successfully.'}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'rlvr-rewards' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-4">
                <Gauge className="w-5 h-5 text-[var(--ds-accent)]" />
                <h2 className="text-md font-bold uppercase tracking-wider">Verifiable Feedback / RLVR Reward Tuner</h2>
              </div>

              {/* Warning flags if weights are bad */}
              {formatWeight > correctWeight * 0.8 && (
                <div className="p-3 bg-amber-50 border border-amber-300 text-amber-950 text-xs rounded flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 shrink-0 text-amber-600" />
                  <div>
                    <strong className="block font-bold">WARNING: Formatting rewards dominate!</strong>
                    The format reward weight is too high compared to correctness. This can lead to format reward hacking (nested tags, empty reasoning).
                  </div>
                </div>
              )}

              {lengthPenalty < 0.0002 && (
                <div className="p-3 bg-red-50 border border-red-300 text-red-950 text-xs rounded flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 shrink-0 text-red-600" />
                  <div>
                    <strong className="block font-bold">WARNING: No length penalty!</strong>
                    With weak length constraint, the model will generate excessively long, redundant traces (overthinking) during policy gradients.
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Sliders panel */}
                <div className="bg-[var(--ds-paper)] p-5 border border-[var(--ds-rule)] rounded space-y-5">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--ds-ink)]">Reward Coefficients</h3>

                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Correctness Weight: {correctWeight}
                    </label>
                    <input
                      type="range"
                      min="0.0"
                      max="2.0"
                      step="0.1"
                      value={correctWeight}
                      onChange={(e) => setCorrectWeight(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Formatting Tag Scorer Weight: {formatWeight}
                    </label>
                    <input
                      type="range"
                      min="0.0"
                      max="1.5"
                      step="0.05"
                      value={formatWeight}
                      onChange={(e) => setFormatWeight(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Language Coherence Weight: {langWeight}
                    </label>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={langWeight}
                      onChange={(e) => setLangWeight(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Token Length Penalty: {lengthPenalty}
                    </label>
                    <input
                      type="range"
                      min="0.0000"
                      max="0.0030"
                      step="0.0001"
                      value={lengthPenalty}
                      onChange={(e) => setLengthPenalty(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>
                </div>

                {/* Score forecast panel */}
                <div className="md:col-span-2 space-y-3">
                  <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Live Reward Forecast per Trace Variant:</span>
                  <div className="space-y-3">
                    {liveRewardsData.map(trace => {
                      const finalReward = trace.liveScores.totalReward;
                      
                      return (
                        <div
                          key={trace.id}
                          className="p-3 border border-[var(--ds-rule)] bg-[var(--ds-paper)] rounded text-xs flex flex-col md:flex-row md:items-center justify-between gap-3"
                        >
                          <div>
                            <span className="font-bold block text-[var(--ds-ink)]">{trace.title}</span>
                            <span className="text-[10px] text-[var(--ds-faint)]">
                              Tokens: {trace.stats.length} | Format: {trace.stats.format} | Correct: {trace.stats.correctness}
                            </span>
                          </div>
                          <div className="flex items-center gap-3">
                            <div className="text-right font-mono text-[10px] text-[var(--ds-faint)] hidden md:block">
                              <span>({trace.liveScores.correctScore.toFixed(2)}C + {trace.liveScores.formatScore.toFixed(2)}F + {trace.liveScores.langScore.toFixed(2)}L - {Math.abs(trace.liveScores.lengthScore).toFixed(2)}Len)</span>
                            </div>
                            <div className="text-right">
                              <span className="block text-[8px] text-[var(--ds-faint)] uppercase">Total Reward</span>
                              <span className={`font-mono font-bold text-sm ${finalReward > 1.0 ? 'text-green-700' : 'text-stone-800'}`}>
                                {finalReward}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'grpo-workbench' && (
          <div className="space-y-6">
            {/* HERO WIDGET: GRPO Reasoning Trainer */}
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 relative overflow-hidden rounded">
              
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--ds-rule)] pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-2.5 w-2.5 rounded-full bg-[var(--ds-ok)] animate-pulse" />
                  <span className="text-xs font-bold tracking-wider uppercase text-[var(--ds-ink)]">
                    GRPO Advantage Policy Trainer Simulator
                  </span>
                </div>
                
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    data-math-control
                    onClick={handleStepGRPO}
                    className="ds-btn font-bold py-1 px-3 text-xs bg-[var(--ds-accent)] hover:bg-stone-800 text-[var(--ds-paper)] transition-all flex items-center gap-1.5"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    RUN POLICY GRADIENT STEP
                  </button>
                  <button
                    data-math-control
                    onClick={() => {
                      setStepCount(0);
                      setGrpoHistory([78, 80, 81, 82, 85, 87]);
                    }}
                    className="ds-btn font-bold py-1 px-3 text-xs border border-[var(--ds-rule)] text-[var(--ds-faint)] hover:text-[var(--ds-ink)] transition-all"
                  >
                    RESET
                  </button>
                </div>
              </div>

              {/* Warnings inside simulator */}
              {grpoCalculations.isAllNegative && (
                <div className="p-3 bg-red-50 border border-red-300 text-red-950 text-xs rounded mb-4 flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 shrink-0 text-red-600" />
                  <div>
                    <strong className="block font-bold">ALL-NEGATIVE GROUP ERROR!</strong>
                    None of the {grpoGroupSize} candidates got correct answers. Advantage calculations will normalize advantages anyway, potentially reinforcing incorrect reasoning pathways!
                  </div>
                </div>
              )}

              {/* Two Column Simulator Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                
                {/* Hyperparameters Controls */}
                <div className="space-y-4 bg-[var(--ds-paper)] p-4 border border-[var(--ds-rule)] rounded">
                  <h3 className="text-xs font-bold uppercase tracking-wider border-b border-[var(--ds-rule)] pb-2 mb-2">Trainer Params</h3>
                  
                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Group Size (G): {grpoGroupSize}
                    </label>
                    <input
                      type="range"
                      min="4"
                      max="8"
                      step="1"
                      value={grpoGroupSize}
                      onChange={(e) => setGrpoGroupSize(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold uppercase text-[var(--ds-faint)] tracking-wider mb-2">
                      Clipping Epsilon (ε): {grpoClipEpsilon}
                    </label>
                    <input
                      type="range"
                      min="0.05"
                      max="0.40"
                      step="0.05"
                      value={grpoClipEpsilon}
                      onChange={(e) => setGrpoClipEpsilon(Number(e.target.value))}
                      className="ds-range"
                    />
                  </div>

                  <div className="border-t border-[var(--ds-rule)] pt-3 space-y-3">
                    <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Active Check Weights:</span>
                    <div>
                      <label className="block text-[9px] font-bold uppercase text-[var(--ds-faint)] mb-1">Correctness: {grpoCorrectWeight}</label>
                      <input
                        type="range"
                        min="0.0"
                        max="1.5"
                        step="0.1"
                        value={grpoCorrectWeight}
                        onChange={(e) => setGrpoCorrectWeight(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                    <div>
                      <label className="block text-[9px] font-bold uppercase text-[var(--ds-faint)] mb-1">Format: {grpoFormatWeight}</label>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={grpoFormatWeight}
                        onChange={(e) => setGrpoFormatWeight(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                    <div>
                      <label className="block text-[9px] font-bold uppercase text-[var(--ds-faint)] mb-1">Len Penalty: {grpoLengthPenalty}</label>
                      <input
                        type="range"
                        min="0.0000"
                        max="0.0020"
                        step="0.0001"
                        value={grpoLengthPenalty}
                        onChange={(e) => setGrpoLengthPenalty(Number(e.target.value))}
                        className="ds-range"
                      />
                    </div>
                  </div>
                </div>

                {/* Candidate advantage table */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="flex items-center justify-between border-b border-[var(--ds-rule)] pb-2">
                    <div>
                      <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Active Query:</span>
                      <p className="text-xs font-mono text-[var(--ds-ink)]">{grpoPromptObj.prompt}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <span className="text-[9px] block font-bold text-[var(--ds-faint)] uppercase">Group Stats</span>
                      <span className="text-xs font-mono font-bold block">μ: {grpoCalculations.mean}</span>
                      <span className="text-xs font-mono font-bold block text-[var(--ds-accent)]">σ: {grpoCalculations.std}</span>
                    </div>
                  </div>

                  <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
                    {grpoCalculations.candidates.map((cand, idx) => {
                      const isPositive = cand.metrics.advantage >= 0;
                      return (
                        <div
                          key={cand.id}
                          className={`p-3 border rounded text-xs transition-all ${
                            isPositive ? 'border-green-200 bg-green-50/20' : 'border-red-200 bg-red-50/20'
                          }`}
                        >
                          <div className="flex items-center justify-between font-mono mb-2">
                            <span className="font-bold text-[var(--ds-ink)]">Candidate {idx + 1}</span>
                            <div className="flex gap-4">
                              <span>Raw Reward: {cand.metrics.totalReward}</span>
                              <span className={isPositive ? 'text-green-700 font-bold' : 'text-red-700 font-bold'}>
                                Adv: {cand.metrics.advantage > 0 ? '+' : ''}{cand.metrics.advantage}
                              </span>
                            </div>
                          </div>
                          
                          <pre className="text-[10px] font-mono bg-stone-900 text-stone-300 p-2 rounded max-h-20 overflow-y-auto whitespace-pre-wrap leading-tight mb-2">
                            {cand.reasoning}
                          </pre>
                          
                          <div className="flex justify-between items-center text-[10px] text-[var(--ds-faint)]">
                            <span>{cand.explanation}</span>
                            <span className={`font-bold ${cand.metrics.isClipped ? 'text-amber-600' : (isPositive ? 'text-green-700' : 'text-red-700')}`}>
                              Update Direction: {cand.metrics.clipUpdate} {cand.metrics.isClipped && '(CLIPPED)'}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Training metrics overview */}
                <div className="space-y-4 bg-[var(--ds-paper)] p-4 border border-[var(--ds-rule)] rounded flex flex-col justify-between">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider border-b border-[var(--ds-rule)] pb-2 mb-2">Policy KPIs</h3>
                    
                    <div className="space-y-3 pt-2">
                      <div className="text-center p-3 bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded">
                        <span className="text-[9px] font-bold text-[var(--ds-faint)] uppercase block">Policy Step</span>
                        <span className="text-xl font-bold font-mono text-[var(--ds-ink)]">{stepCount}</span>
                      </div>
                      
                      <div className="text-center p-3 bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded">
                        <span className="text-[9px] font-bold text-[var(--ds-faint)] uppercase block">Average Accuracy</span>
                        <span className="text-xl font-bold font-mono text-[var(--ds-accent)]">{grpoHistory[grpoHistory.length - 1]}%</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <span className="text-[8px] font-bold uppercase text-[var(--ds-faint)] block mb-1">Accuracy Trend Graph:</span>
                    <div className="h-20 bg-[var(--ds-panel)] border border-[var(--ds-rule)] rounded p-2 flex items-end justify-between gap-1">
                      {grpoHistory.map((val, idx) => (
                        <div key={idx} className="flex-1 flex flex-col items-center gap-1">
                          <div
                            style={{ height: `${(val - 40) * 1.5}px` }}
                            className="w-full bg-[var(--ds-accent)] rounded-t transition-all duration-300"
                          />
                          <span className="text-[8px] font-mono block scale-90">{val}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}

        {activeTab === 'cold-start-vs-rl' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-4">
                <Flame className="w-5 h-5 text-[var(--ds-accent)]" />
                <h2 className="text-md font-bold uppercase tracking-wider">Cold-Start SFT vs Pure RL Training</h2>
              </div>
              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                DeepSeek-R1-Zero proved that reasoning models can learn to think using purely reinforcement learning from verifiable rewards. 
                However, to build structured, readable outputs, DeepSeek-R1 introduces a small cold-start supervised dataset before running RL.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {COLD_START_VS_PURE_RL.map((stage, idx) => (
                  <div key={idx} className="p-5 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded space-y-4">
                    <div className="border-b border-[var(--ds-rule)] pb-2">
                      <h3 className="text-sm font-bold text-[var(--ds-ink)]">{stage.stage}</h3>
                      <span className="text-[10px] text-[var(--ds-faint)] uppercase block mt-1">Dataset Bootstrapping: {stage.dataset}</span>
                    </div>

                    <div className="space-y-2">
                      <span className="text-[10px] font-bold uppercase text-[var(--ds-faint)] block">Observed Training Behaviors:</span>
                      <ul className="space-y-2">
                        {stage.behaviors.map((beh, i) => (
                          <li key={i} className="text-xs text-[var(--ds-ink)] flex items-start gap-2">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--ds-accent)] mt-1.5 shrink-0" />
                            <span>{beh}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="pt-2 grid grid-cols-1 gap-2 text-xs border-t border-[var(--ds-rule)]">
                      <div>
                        <strong className="text-green-700 block font-bold">Pro:</strong>
                        <span className="text-[var(--ds-faint)]">{stage.pros}</span>
                      </div>
                      <div className="pt-1">
                        <strong className="text-red-700 block font-bold">Con:</strong>
                        <span className="text-[var(--ds-faint)]">{stage.cons}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'failure-modes' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Sidebar selector */}
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 rounded space-y-3">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-3 mb-2">
                <AlertTriangle className="w-5 h-5 text-[var(--ds-accent)]" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--ds-ink)]">Reasoning Failure Modes</h3>
              </div>
              <div className="space-y-2">
                {FAILURE_MODES.map(fail => {
                  const isSelected = activeFailureId === fail.id;
                  const isFixed = appliedFixes[fail.id];
                  return (
                    <button
                      key={fail.id}
                      data-math-control
                      onClick={() => setActiveFailureId(fail.id)}
                      className={`w-full text-left p-3 rounded border text-xs transition-all flex items-center justify-between ${
                        isSelected
                          ? 'border-[var(--ds-accent)] bg-[var(--ds-paper)] font-bold'
                          : 'border-[var(--ds-rule)] bg-[var(--ds-paper-2)] hover:bg-[var(--ds-paper)]'
                      }`}
                    >
                      <span>{fail.name}</span>
                      <span className={`inline-block h-2 w-2 rounded-full ${isFixed ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Main Diagnostics Display */}
            <div className="md:col-span-2 border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-5 rounded space-y-5">
              <div className="flex items-center justify-between border-b border-[var(--ds-rule)] pb-3">
                <h3 className="text-sm font-bold text-[var(--ds-ink)]">{currentFailureObj.name} Debugger</h3>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  appliedFixes[currentFailureObj.id] ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800 animate-pulse'
                }`}>
                  {appliedFixes[currentFailureObj.id] ? 'PATCHED' : 'ACTIVE FAILURE RISK'}
                </span>
              </div>

              <div className="space-y-3 text-xs leading-relaxed">
                <div>
                  <strong className="text-[var(--ds-faint)] uppercase text-[9px] block mb-1">Symptom:</strong>
                  <p className="p-3 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded text-[var(--ds-ink)]">{currentFailureObj.symptom}</p>
                </div>
                <div>
                  <strong className="text-[var(--ds-faint)] uppercase text-[9px] block mb-1">Trigger / Root Cause:</strong>
                  <p className="p-3 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded text-[var(--ds-ink)]">{currentFailureObj.trigger}</p>
                </div>
                <div>
                  <strong className="text-[var(--ds-faint)] uppercase text-[9px] block mb-1">Recommended Fix:</strong>
                  <p className="p-3 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded text-[var(--ds-ink)]">{currentFailureObj.fix}</p>
                </div>
              </div>

              <div className="pt-4 border-t border-[var(--ds-rule)] flex justify-end">
                <button
                  data-math-control
                  onClick={() => setAppliedFixes(prev => ({
                    ...prev,
                    [currentFailureObj.id]: !prev[currentFailureObj.id]
                  }))}
                  className={`ds-btn font-bold py-1.5 px-4 text-xs rounded transition-all ${
                    appliedFixes[currentFailureObj.id]
                      ? 'bg-[var(--ds-paper-2)] text-[var(--ds-faint)] border border-[var(--ds-rule)]'
                      : 'bg-green-700 hover:bg-green-800 text-white'
                  }`}
                >
                  {appliedFixes[currentFailureObj.id] ? 'REMOVE PATCH' : 'APPLY BUG PATCH'}
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'distillation' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-4">
                <Sparkles className="w-5 h-5 text-[var(--ds-accent)]" />
                <h2 className="text-md font-bold uppercase tracking-wider">Distillation to Smaller Parameters</h2>
              </div>
              
              <p className="text-xs text-[var(--ds-faint)] leading-relaxed">
                Trace datasets generated by massive frontier models (e.g. DeepSeek-R1, 671B parameters) can be distilled into smaller models. 
                This teaches student parameters (1.5B, 7B, 8B) how to structure their thoughts, although they struggle to generalize on extremely difficult mathematical problems.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {DISTILLATION_PIPELINES.map((step, idx) => (
                  <div key={idx} className="p-4 bg-[var(--ds-paper)] border border-[var(--ds-rule)] rounded flex flex-col justify-between h-44">
                    <div>
                      <h4 className="text-xs font-bold text-[var(--ds-ink)] mb-1">{step.step}</h4>
                      <p className="text-[11px] text-[var(--ds-faint)] leading-relaxed">{step.description}</p>
                    </div>
                    
                    <div className="pt-2 border-t border-[var(--ds-rule)] font-mono text-[9px] text-[var(--ds-accent)]">
                      {step.flow}
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 bg-amber-50 border border-amber-200 text-amber-950 text-xs rounded">
                <strong className="font-bold block mb-1">Production Caveat: The Distillation Generalization Gap</strong>
                While a distilled student model successfully copies the style, planning tags, and backtracking formats of the teacher, 
                its lower parameter count means its raw logic and problem-solving limits remain bounded by its base pretraining capacity.
              </div>
            </div>
          </div>
        )}

        {activeTab === 'paper-decoder' && (
          <div className="space-y-6">
            <div className="border border-[var(--ds-rule)] bg-[var(--ds-panel)] p-6 rounded space-y-6">
              <div className="flex items-center gap-2 border-b border-[var(--ds-rule)] pb-4">
                <Brain className="w-5 h-5 text-[var(--ds-accent)]" />
                <h2 className="text-md font-bold uppercase tracking-wider">Frontier Literature Decoder</h2>
              </div>

              {/* Quote details */}
              <div className="bg-[var(--ds-paper)] p-5 border border-[var(--ds-rule)] rounded space-y-4">
                <div className="border-l-4 border-[var(--ds-accent)] pl-4 italic text-xs text-[var(--ds-ink)] leading-relaxed">
                  "{activeQuote.text}"
                </div>
                <span className="block text-[9px] font-mono text-[var(--ds-faint)] uppercase">SOURCE: DeepSeek-R1 Technical Report</span>
              </div>

              {/* Interactive question card */}
              <div className="bg-[var(--ds-paper)] p-5 border border-[var(--ds-rule)] rounded space-y-4">
                <h3 className="text-xs font-bold text-[var(--ds-ink)]">{activeQuote.question}</h3>
                
                <div className="space-y-2">
                  {activeQuote.choices.map((choice, index) => {
                    const isSelected = selectedPaperAnswer === index;
                    const isCorrect = index === activeQuote.answerIndex;
                    let choiceClass = 'border-[var(--ds-rule)] bg-[var(--ds-paper)] hover:bg-[var(--ds-paper-2)]';
                    
                    if (showPaperExplanation) {
                      if (isCorrect) {
                        choiceClass = 'border-green-300 bg-green-50/50 text-green-900';
                      } else if (isSelected) {
                        choiceClass = 'border-red-300 bg-red-50/50 text-red-900';
                      }
                    } else if (isSelected) {
                      choiceClass = 'border-[var(--ds-accent)] bg-[var(--ds-warm)]';
                    }

                    return (
                      <button
                        key={index}
                        data-math-control
                        disabled={showPaperExplanation}
                        onClick={() => setSelectedPaperAnswer(index)}
                        className={`w-full text-left p-3 rounded border text-xs transition-all flex items-center justify-between ${choiceClass}`}
                      >
                        <span>{choice}</span>
                        {showPaperExplanation && isCorrect && <CheckCircle className="w-4 h-4 text-green-600" />}
                        {showPaperExplanation && isSelected && !isCorrect && <XCircle className="w-4 h-4 text-red-600" />}
                      </button>
                    );
                  })}
                </div>

                <div className="flex justify-between items-center pt-2">
                  <div className="flex gap-2">
                    <button
                      data-math-control
                      onClick={() => {
                        setSelectedPaperQuoteIndex((prev) => (prev === 0 ? 1 : 0));
                        setSelectedPaperAnswer(null);
                        setShowPaperExplanation(false);
                      }}
                      className="ds-btn border border-[var(--ds-rule)] text-[var(--ds-faint)] hover:text-[var(--ds-ink)] text-xs font-bold py-1 px-3 rounded transition-all"
                    >
                      SWITCH PAPER QUOTE
                    </button>
                  </div>
                  
                  <button
                    data-math-control
                    disabled={selectedPaperAnswer === null}
                    onClick={() => setShowPaperExplanation(true)}
                    className="ds-btn bg-[var(--ds-accent)] hover:bg-stone-800 text-[var(--ds-paper)] text-xs font-bold py-1.5 px-4 rounded transition-all disabled:opacity-50"
                  >
                    CHECK ANSWER
                  </button>
                </div>

                {showPaperExplanation && (
                  <div className="p-3 bg-blue-50 border border-blue-200 text-blue-950 text-xs rounded mt-3 leading-relaxed">
                    <strong className="block font-bold">Explanation:</strong>
                    {activeQuote.explanation}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
