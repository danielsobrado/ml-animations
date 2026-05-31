export const CROSS_VALIDATION_ROWS = Object.freeze([
  { id: 'A', user: 'u01', segment: 'new', difficulty: 0.18, target: 0 },
  { id: 'B', user: 'u02', segment: 'new', difficulty: 0.31, target: 0 },
  { id: 'C', user: 'u03', segment: 'returning', difficulty: 0.44, target: 1 },
  { id: 'D', user: 'u04', segment: 'returning', difficulty: 0.52, target: 1 },
  { id: 'E', user: 'u02', segment: 'new', difficulty: 0.34, target: 0 },
  { id: 'F', user: 'u05', segment: 'enterprise', difficulty: 0.67, target: 1 },
  { id: 'G', user: 'u06', segment: 'enterprise', difficulty: 0.73, target: 1 },
  { id: 'H', user: 'u07', segment: 'returning', difficulty: 0.58, target: 1 },
  { id: 'I', user: 'u08', segment: 'new', difficulty: 0.27, target: 0 },
  { id: 'J', user: 'u04', segment: 'returning', difficulty: 0.55, target: 1 },
  { id: 'K', user: 'u09', segment: 'enterprise', difficulty: 0.81, target: 1 },
  { id: 'L', user: 'u10', segment: 'new', difficulty: 0.22, target: 0 },
]);

export function assignFolds(k, strategy, rows = CROSS_VALIDATION_ROWS) {
  return rows.map((row, index) => ({
    ...row,
    fold: strategy === 'grouped' ? assignGroupedFold(row, k) : assignRandomFold(row, index, k),
  }));
}

export function duplicateUserLeakage(rows, validationFold) {
  const trainUsers = new Set(rows.filter((row) => row.fold !== validationFold).map((row) => row.user));
  return rows
    .filter((row) => row.fold === validationFold && trainUsers.has(row.user))
    .map((row) => row.user);
}

export function scoreFold(rows, validationFold, preprocessingInsideFold) {
  const validationRows = rows.filter((row) => row.fold === validationFold);
  const leakageUsers = duplicateUserLeakage(rows, validationFold);
  const segmentPenalty = validationRows.some((row) => row.segment === 'enterprise') ? 0.035 : 0;
  const foldDifficulty =
    validationRows.reduce((sum, row) => sum + row.difficulty, 0) / Math.max(validationRows.length, 1);
  const base = 0.79 - segmentPenalty - Math.abs(foldDifficulty - 0.5) * 0.08;
  const duplicateBoost = leakageUsers.length * 0.035;
  const preprocessingBoost = preprocessingInsideFold ? 0 : 0.045;

  return {
    score: Math.min(0.97, base + duplicateBoost + preprocessingBoost),
    leakageUsers,
    validationSize: validationRows.length,
  };
}

export function summarize(rows, preprocessingInsideFold, k) {
  const folds = Array.from({ length: k }, (_, fold) => scoreFold(rows, fold, preprocessingInsideFold));
  const scores = folds.map((fold) => fold.score);
  const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) / scores.length;
  const leakedFoldCount = folds.filter((fold) => fold.leakageUsers.length > 0).length;

  return {
    folds,
    mean,
    std: Math.sqrt(variance),
    leakedFoldCount,
    min: Math.min(...scores),
    max: Math.max(...scores),
  };
}

function assignRandomFold(row, index, k) {
  return (index * 2 + row.id.charCodeAt(0)) % k;
}

function assignGroupedFold(row, k) {
  const userNumber = Number(row.user.replace('u', ''));
  return userNumber % k;
}
