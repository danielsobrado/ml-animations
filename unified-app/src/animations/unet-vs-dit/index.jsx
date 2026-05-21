import React, { useMemo, useState } from 'react';
import { Boxes, Brain, Grid3X3, SlidersHorizontal } from 'lucide-react';
import AssessmentPanel from '../../components/animation-shell/AssessmentPanel';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function UnetVsDitAnimation() {
  const [architecture, setArchitecture] = useState('unet');
  const [resolution, setResolution] = useState(64);
  const [patchSize, setPatchSize] = useState(8);
  const [depth, setDepth] = useState(6);

  const comparison = useMemo(() => {
    const tokens = Math.round((resolution / patchSize) ** 2);
    const attentionCost = tokens ** 2;
    const unetContext = clamp((depth * 9) + (resolution / 4), 0, 100);
    const ditContext = clamp(35 + depth * 8 + (tokens / 10), 0, 100);
    const memoryPressure = architecture === 'dit'
      ? clamp(attentionCost / 110, 8, 100)
      : clamp((resolution * depth) / 12, 8, 100);
    const localBias = architecture === 'unet' ? 92 : clamp(80 - tokens / 5, 25, 85);
    const globalMixing = architecture === 'dit' ? ditContext : unetContext;

    return {
      tokens,
      attentionCost,
      localBias,
      globalMixing,
      memoryPressure,
    };
  }, [architecture, depth, patchSize, resolution]);

  const cells = Array.from({ length: 16 }, (_, index) => index);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <SlidersHorizontal size={16} />
            Architecture controls
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            {[
              { id: 'unet', label: 'U-Net', icon: Grid3X3, copy: 'Convolutional pyramid with skip connections.' },
              { id: 'dit', label: 'DiT', icon: Brain, copy: 'Transformer over image or latent patches.' },
            ].map((option) => {
              const Icon = option.icon;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setArchitecture(option.id)}
                  className={`rounded-xl border p-4 text-left transition ${
                    architecture === option.id
                      ? 'border-slate-900 bg-slate-900 text-white'
                      : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-fuchsia-300'
                  }`}
                >
                  <span className="flex items-center gap-2 font-semibold">
                    <Icon size={16} />
                    {option.label}
                  </span>
                  <span className="mt-2 block text-sm opacity-80">{option.copy}</span>
                </button>
              );
            })}
          </div>

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="uvd-resolution">
            Latent resolution: {resolution} x {resolution}
          </label>
          <input
            id="uvd-resolution"
            type="range"
            min="32"
            max="128"
            step="16"
            value={resolution}
            onChange={(event) => setResolution(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="uvd-patch">
            DiT patch size: {patchSize}
          </label>
          <input
            id="uvd-patch"
            type="range"
            min="4"
            max="16"
            step="4"
            value={patchSize}
            onChange={(event) => setPatchSize(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />

          <label className="mt-5 block text-sm font-semibold text-slate-700" htmlFor="uvd-depth">
            Network depth: {depth}
          </label>
          <input
            id="uvd-depth"
            type="range"
            min="3"
            max="12"
            step="1"
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value))}
            className="mt-2 w-full accent-fuchsia-500"
          />
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-fuchsia-700">
            <Boxes size={16} />
            Feature geometry
          </div>

          <div className="grid gap-4 md:grid-cols-[0.8fr_1.2fr]">
            <div className="grid grid-cols-4 gap-2 rounded-xl border border-slate-200 bg-slate-50 p-4">
              {cells.map((cell) => (
                <div
                  key={cell}
                  className={`aspect-square rounded-lg ${
                    architecture === 'unet'
                      ? cell % 5 === 0 ? 'bg-fuchsia-500' : 'bg-slate-300'
                      : 'bg-sky-400'
                  }`}
                />
              ))}
            </div>
            <div className="space-y-3">
              {[
                { label: 'Local image bias', value: comparison.localBias, color: 'bg-fuchsia-500' },
                { label: 'Global mixing', value: comparison.globalMixing, color: 'bg-sky-500' },
                { label: 'Memory pressure', value: comparison.memoryPressure, color: 'bg-amber-500' },
              ].map((metric) => (
                <div key={metric.label}>
                  <div className="mb-1 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <span>{metric.label}</span>
                    <span>{metric.value.toFixed(0)}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-slate-100">
                    <div className={`h-full rounded-full ${metric.color}`} style={{ width: `${metric.value}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">DiT tokens</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{comparison.tokens}</div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">Attention pairs</div>
              <div className="mt-2 text-3xl font-bold text-slate-900">{comparison.attentionCost}</div>
            </div>
            <div className="rounded-xl border border-slate-900 bg-slate-900 p-4 text-white">
              <div className="text-xs uppercase tracking-wide text-fuchsia-200">Best instinct</div>
              <div className="mt-2 text-xl font-bold">{architecture === 'unet' ? 'local pyramid' : 'global patches'}</div>
            </div>
          </div>

          <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            U-Nets bring strong local image structure and efficient downsample-upsample paths; DiTs trade that built-in bias for scalable global attention over latent patches.
          </p>
        </div>
      </section>

      <AssessmentPanel lessonId="unet-vs-dit" title="U-Net vs DiT check" />
    </div>
  );
}
