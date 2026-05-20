import fs from 'fs';
import path from 'path';

const root = process.cwd();
const unifiedAnimations = path.join(root, 'unified-app', 'src', 'animations');

const files = [];

if (fs.existsSync(unifiedAnimations)) {
  for (const entry of fs.readdirSync(unifiedAnimations, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const indexFile = path.join(unifiedAnimations, entry.name, 'index.jsx');
    if (fs.existsSync(indexFile)) files.push(indexFile);
  }
}

for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
  if (!entry.isDirectory() || !entry.name.endsWith('-animation')) continue;
  for (const candidate of ['src/App.jsx', 'src/index.jsx']) {
    const file = path.join(root, entry.name, candidate);
    if (fs.existsSync(file)) files.push(file);
  }
}

let changed = 0;

for (const file of files) {
  const source = fs.readFileSync(file, 'utf8');
  const next = source.replace(/,\s*color:\s*['"`]from-[^'"`]+? to-[^'"`]+?['"`]/g, '');
  if (next !== source) {
    fs.writeFileSync(file, next);
    changed += 1;
  }
}

console.log(`Removed unused tab gradient color metadata from ${changed} files.`);
