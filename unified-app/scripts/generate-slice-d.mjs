import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MAPS as MAPS1 } from './slice-d-maps-part1.mjs';
import { MAPS as MAPS2 } from './slice-d-maps-part2.mjs';
import { MAPS as MAPS3 } from './slice-d-maps-part3.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = join(__dirname, '../src/data/_concept-maps-40-slice-d.js');
const MAPS = [...MAPS1, ...MAPS2, ...MAPS3];

function escapeStr(value) {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r\n/g, '\n')
    .replace(/\n/g, '\\n');
}

function renderLeaf(node) {
  const fields = Object.entries(node.tip)
    .map(([key, value]) => `              ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const lesson = node.lessonId ? `\n            lessonId: '${node.lessonId}',` : '';
  return `          {
            id: '${node.id}',
            label: '${escapeStr(node.label)}',
            tooltip: tip({
${fields}
            }),${lesson}
          }`;
}

function renderBranch(branch) {
  const children = branch.children.map(renderLeaf).join(',\n');
  return `      {
        id: '${branch.id}',
        label: '${branch.label}',
        type: '${branch.type}',
        children: [
${children}
        ],
      }`;
}

function renderMap(map) {
  const centerFields = Object.entries(map.center)
    .map(([key, value]) => `        ${key}: '${escapeStr(value)}',`)
    .join('\n');
  const branches = map.branches.map(renderBranch).join(',\n');
  return `  '${map.id}': {
    center: {
      id: '${map.id}',
      label: '${escapeStr(map.label)}',
      type: 'current',
      tooltip: tip({
${centerFields}
      }),
    },
    branches: [
${branches}
    ],
  }`;
}

writeFileSync(
  outPath,
  `function tip(fields) {
  return fields;
}

export const SLICE_D = {
${MAPS.map(renderMap).join(',\n')}
};
`,
  'utf8',
);

for (const map of MAPS) {
  let leaves = 0;
  for (const branch of map.branches) leaves += branch.children.length;
  console.log(`${map.id}: ${leaves} leaves`);
}
console.log('Total maps:', MAPS.length);
console.log('Wrote', outPath);
