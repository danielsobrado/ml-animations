export const LEAKAGE_MODES = Object.freeze({
  duplicates: {
    label: 'Duplicate users',
    leak: 'Rows from the same user appear in both train and validation.',
    fix: 'Group by user before splitting so all rows for one user stay together.',
    suspiciousScore: 0.93,
    honestScore: 0.72,
    leakedItem: 'user_104',
  },
  preprocessing: {
    label: 'Preprocessing before split',
    leak: 'Scaler, imputer, or feature selector learns validation statistics.',
    fix: 'Fit preprocessing inside each training fold, then transform validation with training parameters.',
    suspiciousScore: 0.9,
    honestScore: 0.75,
    leakedItem: 'global mean',
  },
  target: {
    label: 'Target leakage',
    leak: 'A feature is created from information that would only be known after the prediction time.',
    fix: 'Remove target-derived and future-known features from the training schema.',
    suspiciousScore: 0.98,
    honestScore: 0.68,
    leakedItem: 'post_outcome_code',
  },
  time: {
    label: 'Time leakage',
    leak: 'Random splits let future rows teach a model evaluated on past rows.',
    fix: 'Use chronological splits and only train on data available before the validation window.',
    suspiciousScore: 0.91,
    honestScore: 0.7,
    leakedItem: 'future row',
  },
  testTuning: {
    label: 'Repeated test tuning',
    leak: 'The test set is checked after every modeling decision until it becomes development feedback.',
    fix: 'Use validation for iteration and touch the test set once for final reporting.',
    suspiciousScore: 0.88,
    honestScore: 0.73,
    leakedItem: 'test score',
  },
});

export const LEAKAGE_ROWS = Object.freeze([
  { id: 'A', user: 'user_101', time: 'Jan', split: 'train', target: 0 },
  { id: 'B', user: 'user_104', time: 'Feb', split: 'train', target: 1 },
  { id: 'C', user: 'user_118', time: 'Mar', split: 'train', target: 0 },
  { id: 'D', user: 'user_104', time: 'Apr', split: 'validation', target: 1 },
  { id: 'E', user: 'user_132', time: 'May', split: 'validation', target: 0 },
  { id: 'F', user: 'user_150', time: 'Jun', split: 'test', target: 1 },
]);

export function scoreGap(mode, strictPipeline) {
  const config = LEAKAGE_MODES[mode];
  const suspicious = strictPipeline ? config.honestScore : config.suspiciousScore;
  const honest = config.honestScore;
  return {
    suspicious,
    honest,
    optimism: Math.max(0, suspicious - honest),
  };
}

export function rowIsLeaked(row, mode) {
  if (mode === 'duplicates') return row.user === 'user_104';
  if (mode === 'time') return row.time === 'Jun' || row.time === 'May';
  if (mode === 'target') return row.id === 'B' || row.id === 'D' || row.id === 'F';
  if (mode === 'preprocessing') return row.split !== 'train';
  return row.split === 'test';
}
