export const TRAIN_VALIDATION_ROWS = Object.freeze([
  { id: '01', time: 1, segment: 'A', y: 0, x: 12 },
  { id: '02', time: 2, segment: 'B', y: 0, x: 18 },
  { id: '03', time: 3, segment: 'A', y: 0, x: 20 },
  { id: '04', time: 4, segment: 'C', y: 1, x: 42 },
  { id: '05', time: 5, segment: 'B', y: 0, x: 24 },
  { id: '06', time: 6, segment: 'A', y: 1, x: 45 },
  { id: '07', time: 7, segment: 'C', y: 1, x: 52 },
  { id: '08', time: 8, segment: 'B', y: 0, x: 28 },
  { id: '09', time: 9, segment: 'A', y: 1, x: 56 },
  { id: '10', time: 10, segment: 'C', y: 1, x: 61 },
  { id: '11', time: 11, segment: 'B', y: 0, x: 33 },
  { id: '12', time: 12, segment: 'A', y: 1, x: 66 },
  { id: '13', time: 13, segment: 'C', y: 1, x: 72 },
  { id: '14', time: 14, segment: 'B', y: 0, x: 35 },
  { id: '15', time: 15, segment: 'A', y: 1, x: 77 },
]);

const BUCKETS = ['train', 'validation', 'test'];

export function splitCounts(total, validationPercent, testPercent) {
  const test = Math.max(1, Math.round(total * testPercent));
  const validation = Math.max(1, Math.round(total * validationPercent));
  const train = Math.max(1, total - validation - test);
  return { train, validation, test };
}

export function assignByMode(mode, validationPercent, testPercent, rows = TRAIN_VALIDATION_ROWS) {
  const counts = splitCounts(rows.length, validationPercent, testPercent);

  if (mode === 'time') {
    return splitInOrder([...rows].sort((a, b) => a.time - b.time), counts);
  }

  if (mode === 'stratified') {
    return stratifiedSplit(rows, counts);
  }

  const shuffled = [...rows].sort((a, b) => randomRank(a) - randomRank(b));
  return splitInOrder(shuffled, counts);
}

export function positiveRate(rows) {
  if (!rows.length) return 0;
  return rows.filter((row) => row.y === 1).length / rows.length;
}

export function meanX(rows) {
  if (!rows.length) return 0;
  return rows.reduce((sum, row) => sum + row.x, 0) / rows.length;
}

export function driftGap(trainRows, targetRows) {
  return Math.abs(meanX(targetRows) - meanX(trainRows));
}

function splitInOrder(rows, counts) {
  return {
    train: rows.slice(0, counts.train),
    validation: rows.slice(counts.train, counts.train + counts.validation),
    test: rows.slice(counts.train + counts.validation),
  };
}

function stratifiedSplit(rows, counts) {
  const buckets = { train: [], validation: [], test: [] };
  const remaining = { ...counts };
  const groups = [...new Set(rows.map((row) => row.y))]
    .sort((a, b) => a - b)
    .map((label) => rows.filter((row) => row.y === label));

  groups.forEach((group, groupIndex) => {
    const allocation = groupIndex === groups.length - 1
      ? { ...remaining }
      : allocateGroup(group.length, rows.length, counts, remaining);

    for (const bucket of BUCKETS) {
      const assigned = group.splice(0, allocation[bucket]);
      buckets[bucket].push(...assigned);
      remaining[bucket] -= assigned.length;
    }
  });

  return buckets;
}

function allocateGroup(groupSize, totalSize, counts, remaining) {
  const allocation = Object.fromEntries(BUCKETS.map((bucket) => [bucket, 0]));
  const targets = BUCKETS.map((bucket) => {
    const target = (groupSize * counts[bucket]) / totalSize;
    const floor = Math.min(Math.floor(target), remaining[bucket]);
    allocation[bucket] = floor;
    return { bucket, remainder: target - floor };
  });

  let leftover = groupSize - BUCKETS.reduce((sum, bucket) => sum + allocation[bucket], 0);
  const ranked = targets.sort((a, b) => {
    if (b.remainder !== a.remainder) return b.remainder - a.remainder;
    return remaining[b.bucket] - remaining[a.bucket];
  });

  while (leftover > 0) {
    const target = ranked.find(({ bucket }) => allocation[bucket] < remaining[bucket]);
    if (!target) break;
    allocation[target.bucket] += 1;
    leftover -= 1;
  }

  return allocation;
}

function randomRank(row) {
  return (Number(row.id) * 7 + 3) % 17;
}
