/**
 * Merges SLICE_A..D fragment files into conceptMaps.js (new keys only).
 */
import { readFileSync, writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataDir = join(__dirname, '../src/data');
const conceptMapsPath = join(dataDir, 'conceptMaps.js');
const sliceFiles = ['a', 'b', 'c', 'd'].map((letter) => ({
  path: join(dataDir, `_concept-maps-40-slice-${letter}.js`),
  exportName: `SLICE_${letter.toUpperCase()}`,
}));

function extractMapKeys(source) {
  const blockMatch = source.match(/export const CONCEPT_MAPS = \{([\s\S]*)\r?\n\};/);
  if (!blockMatch) throw new Error('Could not locate CONCEPT_MAPS block');
  const keys = [];
  const re = /^\s*'([^']+)':\s*\{/gm;
  let match = re.exec(blockMatch[1]);
  while (match) {
    keys.push(match[1]);
    match = re.exec(blockMatch[1]);
  }
  return keys;
}

function renderMapEntry(key, map) {
  const lines = [`  '${key}': {`];
  lines.push('    center: {');
  lines.push(`      id: '${map.center.id}',`);
  lines.push(`      label: ${JSON.stringify(map.center.label)},`);
  lines.push(`      type: '${map.center.type}',`);
  lines.push('      tooltip: tip({');
  for (const [field, value] of Object.entries(map.center.tooltip)) {
    lines.push(`        ${field}: ${JSON.stringify(value)},`);
  }
  lines.push('      }),');
  lines.push('    },');
  lines.push('    branches: [');
  for (const branch of map.branches) {
    lines.push('      {');
    lines.push(`        id: '${branch.id}',`);
    lines.push(`        label: ${JSON.stringify(branch.label)},`);
    lines.push(`        type: '${branch.type}',`);
    lines.push('        children: [');
    for (const child of branch.children) {
      lines.push('          {');
      lines.push(`            id: '${child.id}',`);
      lines.push(`            label: ${JSON.stringify(child.label)},`);
      lines.push('            tooltip: tip({');
      for (const [field, value] of Object.entries(child.tooltip)) {
        lines.push(`              ${field}: ${JSON.stringify(value)},`);
      }
      lines.push('            }),');
      if (child.lessonId) lines.push(`            lessonId: '${child.lessonId}',`);
      lines.push('          },');
    }
    lines.push('        ],');
    lines.push('      },');
  }
  lines.push('    ],');
  lines.push('  },');
  return lines.join('\n');
}

const existingKeys = new Set(extractMapKeys(readFileSync(conceptMapsPath, 'utf8')));
const mergedKeys = [];
const skippedKeys = [];

for (const { path, exportName } of sliceFiles) {
  const module = await import(`${pathToFileURL(path).href}?v=${Date.now()}`);
  const slice = module[exportName];
  for (const [key, map] of Object.entries(slice)) {
    if (existingKeys.has(key)) {
      skippedKeys.push(key);
      continue;
    }
    existingKeys.add(key);
    mergedKeys.push({ key, map });
  }
}

if (!mergedKeys.length) {
  console.log(JSON.stringify({ merged: 0, skipped: skippedKeys }));
  process.exit(0);
}

const insertion = mergedKeys.map(({ key, map }) => renderMapEntry(key, map)).join('\n\n');
let conceptSource = readFileSync(conceptMapsPath, 'utf8');
const replaced = conceptSource.replace(
  /\r?\n};\r?\n\r?\nexport function getConceptMap/,
  `\n\n${insertion}\n};\n\nexport function getConceptMap`,
);
if (replaced === conceptSource) {
  throw new Error('Could not find CONCEPT_MAPS closing block to merge into');
}
writeFileSync(conceptMapsPath, replaced);

for (const { path } of sliceFiles) {
  if (existsSync(path)) unlinkSync(path);
}

console.log(
  JSON.stringify({
    merged: mergedKeys.length,
    mergedKeys: mergedKeys.map(({ key }) => key),
    skipped: skippedKeys,
  }),
);
