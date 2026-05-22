import React, { useEffect, useMemo, useState } from 'react';
import { Activity, Database, Layers3, Network, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

const KV_OPTIONS = [1, 2, 4, 8, 16, 32];
const QUERY_OPTIONS = [8, 16, 32];
const HEAD_DIM_OPTIONS = [64, 128];
const GROUP_COLORS = [
  'bg-sky-500',
  'bg-emerald-500',
  'bg-amber-500',
  'bg-rose-500',
  'bg-violet-500',
  'bg-cyan-500',
  'bg-lime-500',
  'bg-fuchsia-500',
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const formatNumber = (value) => new Intl.NumberFormat('en-US').format(Math.round(value));

function modeLabel(queryHeads, kvHeads) {
  if (kvHeads === queryHeads) return 'Multi-head attention';
  if (kvHeads === 1) return 'Multi-query attention';
  return 'Grouped-query attention';
}

function ControlButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-sm font-semibold transition ${
        active
          ? 'border-slate-900 bg-slate-900 text-white shadow-sm'
          : 'border-slate-200 bg-white text-slate-700 hover:border-slate-400'
      }`}
    >
      {children}
    </button>
  );
}

function MetricCard({ icon: Icon, label, value, helper }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <Icon size={16} />
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
      <p className="mt-1 text-sm text-slate-600">{helper}</p>
    </div>
  );
}

export default function GroupedQueryAttentionAnimation() {
  const [queryHeads, setQueryHeads] = useState(16);
  const [kvHeads, setKvHeads] = useState(4);
  const [sequenceLength, setSequenceLength] = useState(4096);
  const [headDim, setHeadDim] = useState(128);

  const validKvOptions = KV_OPTIONS.filter((heads) => heads <= queryHeads && queryHeads % heads === 0);

  const stats = useMemo(() => {
    const safeKvHeads = validKvOptions.includes(kvHeads) ? kvHeads : validKvOptions[validKvOptions.length - 1];
    const groupSize = queryHeads / safeKvHeads;
    const kvElements = sequenceLength * safeKvHeads * headDim * 2;
    const mhaElements = sequenceLength * queryHeads * headDim * 2;
    const memoryRatio = kvElements / mhaElements;
    const savedPercent = (1 - memoryRatio) * 100;
    const specialization = clamp(100 - (groupSize - 1) * 7, 35, 100);

    const groups = Array.from({ length: safeKvHeads }, (_, groupIndex) => {
      const start = groupIndex * groupSize;
      return {
        id: groupIndex,
        color: GROUP_COLORS[groupIndex % GROUP_COLORS.length],
        kvLabel: `KV ${groupIndex + 1}`,
        queryHeads: Array.from({ length: groupSize }, (_, offset) => start + offset + 1),
      };
    });

    return {
      safeKvHeads,
      groupSize,
      kvElements,
      mhaElements,
      memoryRatio,
      savedPercent,
      specialization,
      groups,
      label: modeLabel(queryHeads, safeKvHeads),
    };
  }, [headDim, kvHeads, queryHeads, sequenceLength, validKvOptions]);

  useEffect(() => {
    if (stats.safeKvHeads !== kvHeads) {
      setKvHeads(stats.safeKvHeads);
    }
  }, [kvHeads, stats.safeKvHeads]);

  return (
    <div className="min-h-full bg-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 md:p-6">
        <header className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
                <Network size={17} />
                Transformer inference
              </div>
              <h1 className="mt-2 text-2xl font-bold text-slate-950 md:text-3xl">
                Grouped-query attention
              </h1>
              <p className="mt-2 max-w-3xl text-slate-700">
                Compare MHA, MQA, and GQA by changing how many key/value heads serve the query heads. The tradeoff is
                smaller KV cache and bandwidth at the cost of less per-head specialization.
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
              <div className="font-semibold text-slate-950">{stats.label}</div>
              <div>
                {queryHeads} query heads share {stats.safeKvHeads} KV heads
              </div>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <aside className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 font-semibold text-slate-950">
              <SlidersHorizontal size={18} />
              Controls
            </div>

            <div className="mt-5 space-y-5">
              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Query heads</div>
                <div className="grid grid-cols-3 gap-2">
                  {QUERY_OPTIONS.map((heads) => (
                    <ControlButton
                      key={heads}
                      active={queryHeads === heads}
                      onClick={() => {
                        setQueryHeads(heads);
                        if (kvHeads > heads || heads % kvHeads !== 0) {
                          setKvHeads(heads);
                        }
                      }}
                    >
                      {heads}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">KV heads</div>
                <div className="grid grid-cols-3 gap-2">
                  {validKvOptions.map((heads) => (
                    <ControlButton key={heads} active={stats.safeKvHeads === heads} onClick={() => setKvHeads(heads)}>
                      {heads}
                    </ControlButton>
                  ))}
                </div>
              </div>

              <label className="block">
                <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                  <span>Context tokens</span>
                  <span>{formatNumber(sequenceLength)}</span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="8192"
                  step="512"
                  value={sequenceLength}
                  onChange={(event) => setSequenceLength(Number(event.target.value))}
                  className="w-full accent-slate-900"
                />
              </label>

              <div>
                <div className="mb-2 text-sm font-semibold text-slate-700">Head dimension</div>
                <div className="grid grid-cols-2 gap-2">
                  {HEAD_DIM_OPTIONS.map((dim) => (
                    <ControlButton key={dim} active={headDim === dim} onClick={() => setHeadDim(dim)}>
                      {dim}
                    </ControlButton>
                  ))}
                </div>
              </div>
            </div>
          </aside>

          <main className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <MetricCard
                icon={Database}
                label="KV cache elements"
                value={formatNumber(stats.kvElements)}
                helper={`${stats.savedPercent.toFixed(0)}% fewer than full MHA for the same context.`}
              />
              <MetricCard
                icon={Activity}
                label="Read bandwidth"
                value={`${(stats.memoryRatio * 100).toFixed(0)}%`}
                helper="Attention still reads cached positions, but fewer K/V heads are stored and read."
              />
              <MetricCard
                icon={Layers3}
                label="Sharing ratio"
                value={`${stats.groupSize}:1`}
                helper="Query heads per shared key/value head."
              />
            </div>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-bold text-slate-950">Query-to-KV grouping</h2>
                  <p className="text-sm text-slate-600">
                    Each row shows one KV head and the query heads that reuse it during decoding.
                  </p>
                </div>
                <div className="rounded-md bg-slate-100 px-3 py-2 text-sm font-semibold text-slate-700">
                  Full MHA cache: {formatNumber(stats.mhaElements)} elements
                </div>
              </div>

              <div className="mt-4 space-y-3">
                {stats.groups.map((group) => (
                  <div key={group.id} className="grid gap-3 rounded-lg border border-slate-200 p-3 md:grid-cols-[90px_1fr]">
                    <div className="flex items-center gap-2 font-semibold text-slate-800">
                      <span className={`h-3 w-3 rounded-full ${group.color}`} />
                      {group.kvLabel}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {group.queryHeads.map((head) => (
                        <span
                          key={head}
                          className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-sm font-medium text-slate-700"
                        >
                          Q{head}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Predict before running</h3>
                <p className="mt-2 text-sm text-slate-700">
                  Lower KV heads should reduce cache size almost linearly with the KV-head ratio. The query count stays
                  the same, so this is not fewer attention heads overall.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Failure mode</h3>
                <p className="mt-2 text-sm text-slate-700">
                  If too many query heads share one KV head, distinct attention patterns can collapse into a shared
                  representation. MQA is cheapest, but usually gives the least specialization.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <h3 className="font-bold text-slate-950">Practical rule</h3>
                <p className="mt-2 text-sm text-slate-700">
                  GQA is the middle ground: keep several KV groups instead of one, while still shrinking long-context
                  cache memory and decode bandwidth.
                </p>
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between text-sm font-semibold text-slate-700">
                <span>Specialization proxy</span>
                <span>{stats.specialization.toFixed(0)}%</span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full bg-slate-900" style={{ width: `${stats.specialization}%` }} />
              </div>
              <p className="mt-2 text-sm text-slate-600">
                This proxy is intentionally qualitative: real quality depends on training and architecture, but the
                causal tradeoff is that fewer KV heads force more sharing.
              </p>
            </section>
          </main>
        </section>

        <AssessmentPanel lessonId="grouped-query-attention" />
      </div>
    </div>
  );
}
