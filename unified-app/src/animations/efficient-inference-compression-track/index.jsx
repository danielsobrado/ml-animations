import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'efficient-inference-compression-track',
  kicker: 'Deployment efficiency',
  title: 'Efficient Inference & Compression',
  description: 'Efficient LLM deployment balances quantization, pruning, distillation, batching, speculative decoding, paged attention, throughput, latency, and memory bandwidth.',
  controls: [
    { id: 'compression', label: 'Compression level', min: 0, max: 100, step: 5, defaultValue: 45, format: pct, help: 'Quantization, pruning, or distillation pressure.' },
    { id: 'batching', label: 'Batching intensity', min: 0, max: 100, step: 5, defaultValue: 55, format: pct, help: 'More batching improves throughput but can add queueing latency.' },
    { id: 'memoryPressure', label: 'Memory pressure', min: 0, max: 100, step: 5, defaultValue: 60, format: pct, help: 'KV cache, model weights, and bandwidth constraints.' },
  ],
  compute(values) {
    const throughput = Math.min(100, 35 + values.compression * 0.35 + values.batching * 0.45 - values.memoryPressure * 0.15);
    const latencyRisk = Math.min(100, values.batching * 0.35 + values.memoryPressure * 0.45 - values.compression * 0.2);
    const qualityRisk = values.compression * 0.45;
    return {
      stats: [
        { label: 'Throughput gain', value: pct(throughput), detail: 'Tokens per second proxy', tone: throughput > 60 ? 'emerald' : 'amber' },
        { label: 'Latency risk', value: pct(latencyRisk), detail: 'Queue and memory pressure', tone: latencyRisk > 45 ? 'rose' : 'cyan' },
        { label: 'Quality risk', value: pct(qualityRisk), detail: 'Compression tradeoff', tone: qualityRisk > 35 ? 'amber' : 'emerald' },
        { label: 'Memory pressure', value: pct(values.memoryPressure), detail: 'Bandwidth and KV cache', tone: values.memoryPressure > 60 ? 'rose' : 'cyan' },
      ],
      bars: [
        { label: 'Throughput improvement', value: pct(throughput), width: throughput, color: 'bg-emerald-500' },
        { label: 'Latency risk', value: pct(latencyRisk), width: latencyRisk, color: 'bg-rose-500' },
        { label: 'Compression quality risk', value: pct(qualityRisk), width: qualityRisk, color: 'bg-amber-500' },
      ],
      formulaLines: [
        'latency target: time to first token + decode time',
        'throughput target: requests/sec or tokens/sec',
        'memory: weights + KV cache + activation workspace',
      ],
      readout: 'The fastest system is not always the best product system. Throughput, latency, quality, and memory fight each other.',
      steps: [
        { title: 'Compress carefully', pass: qualityRisk <= 35, body: qualityRisk <= 35 ? 'Compression risk is moderate.' : 'Aggressive compression needs quality and safety evals.' },
        { title: 'Batch for throughput', pass: throughput >= 55, body: throughput >= 55 ? 'Batching and compression improve serving capacity.' : 'Throughput gains are limited under this setup.' },
        { title: 'Control memory bottlenecks', pass: values.memoryPressure <= 60, body: values.memoryPressure <= 60 ? 'Memory pressure is manageable.' : 'Paged attention, KV-cache policy, or smaller models may be needed.' },
      ],
      takeaway: 'Inference optimization is a Pareto problem. Track tokens/sec, time-to-first-token, memory, and quality together.',
    };
  },
};

export default function EfficientInferenceCompressionTrackAnimation() {
  return <CausalConceptLesson config={config} />;
}
