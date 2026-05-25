import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MAPS_PART1 } from './slice-c-maps-part1.mjs';
import { MAPS_PART2 } from './slice-c-maps-part2.mjs';
import { renderMap } from './slice-c-render.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = join(__dirname, '../src/data/_concept-maps-40-slice-c.js');

const MAPS = [...MAPS_PART1, ...MAPS_PART2];
const body = MAPS.map(renderMap).join(',\n');

writeFileSync(
  outPath,
  `function tip(fields) {
  return fields;
}

export const SLICE_C = {
${body},
};
`,
  'utf8',
);

for (const map of MAPS) {
  const leaves = map.branches.reduce((n, b) => n + b.children.length, 0);
  console.log(`${map.id}: branches=${map.branches.length} leaves=${leaves}`);
}

console.log('Wrote', outPath);
