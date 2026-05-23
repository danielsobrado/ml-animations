import React from 'react';
import CausalConceptLesson from '../_shared/CausalConceptLesson';

const pct = (value) => `${value.toFixed(0)}%`;

const config = {
  lessonId: 'recommender-systems-ranking-track',
  kicker: 'Ranking systems',
  title: 'Recommender Systems & Ranking',
  description: 'Recommendation systems combine collaborative signals, embeddings, implicit feedback, cold-start handling, learning-to-rank losses, ranking metrics, and exploration.',
  controls: [
    { id: 'collab', label: 'Collaborative signal', min: 0, max: 100, step: 5, defaultValue: 55, format: pct, help: 'User-item interaction history available for collaborative filtering.' },
    { id: 'coldStart', label: 'Cold-start pressure', min: 0, max: 100, step: 5, defaultValue: 35, format: pct, help: 'New users or items with little interaction history.' },
    { id: 'exploration', label: 'Exploration budget', min: 0, max: 30, step: 1, defaultValue: 8, format: pct, help: 'Traffic reserved for learning beyond current top-ranked items.' },
  ],
  compute(values) {
    const rankingStrength = Math.min(100, values.collab + values.exploration * 0.8 - values.coldStart * 0.45);
    const coverage = Math.max(0, 100 - values.coldStart + values.exploration);
    return {
      stats: [
        { label: 'Ranking strength', value: pct(rankingStrength), detail: 'Signal minus cold-start drag', tone: rankingStrength > 55 ? 'emerald' : 'amber' },
        { label: 'Catalog coverage', value: pct(coverage), detail: 'Long-tail visibility', tone: coverage > 65 ? 'emerald' : 'cyan' },
        { label: 'Cold-start risk', value: pct(values.coldStart), detail: 'Need content priors', tone: values.coldStart > 50 ? 'rose' : 'amber' },
        { label: 'Explore traffic', value: pct(values.exploration), detail: 'Learning budget', tone: values.exploration > 5 ? 'emerald' : 'amber' },
      ],
      bars: [
        { label: 'Collaborative filtering signal', value: pct(values.collab), width: values.collab, color: 'bg-emerald-500' },
        { label: 'Cold-start pressure', value: pct(values.coldStart), width: values.coldStart, color: 'bg-rose-500' },
        { label: 'Ranking system readiness', value: pct(rankingStrength), width: rankingStrength, color: 'bg-cyan-500' },
      ],
      formulaLines: [
        'matrix factorization: user_vector dot item_vector',
        'ranking metrics: MAP, MRR, nDCG',
        'losses: pointwise, pairwise, listwise',
      ],
      readout: 'Ranking quality is not just prediction accuracy. The ordering, exposure, and feedback loop determine product behavior.',
      steps: [
        { title: 'Use interaction signal', pass: values.collab >= 40, body: values.collab >= 40 ? 'Collaborative history can support personalized ranking.' : 'Sparse interactions require content or popularity baselines.' },
        { title: 'Handle cold start', pass: values.coldStart <= 45, body: values.coldStart <= 45 ? 'Cold-start pressure is manageable.' : 'New users or items need metadata, onboarding, or hybrid retrieval.' },
        { title: 'Explore deliberately', pass: values.exploration >= 5, body: values.exploration >= 5 ? 'Exploration creates data for better future recommendations.' : 'Pure exploitation can trap the system in stale feedback.' },
      ],
      takeaway: 'A recommender is a ranking-and-feedback system. Measure ordered results with ranking metrics and reserve room to learn.',
    };
  },
};

export default function RecommenderSystemsRankingTrackAnimation() {
  return <CausalConceptLesson config={config} />;
}
