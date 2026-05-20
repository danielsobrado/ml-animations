import fs from 'fs';
import path from 'path';

const root = process.cwd();

function sourceRoots() {
  const roots = [path.join(root, 'unified-app', 'src')];

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.endsWith('-animation')) continue;
    const src = path.join(root, entry.name, 'src');
    if (fs.existsSync(src)) roots.push(src);
  }

  return roots;
}

function visit(dir, files = []) {
  if (!fs.existsSync(dir)) return files;

  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== 'dist') visit(file, files);
      continue;
    }

    if (/\.(jsx|tsx|js|ts)$/.test(entry.name)) files.push(file);
  }

  return files;
}

let changed = 0;

for (const rootDir of sourceRoots()) {
  for (const file of visit(rootDir)) {
    const source = fs.readFileSync(file, 'utf8');
    const next = source.replace(/\s+dark:[^\s'"`{}]+/g, '');

    if (next !== source) {
      fs.writeFileSync(file, next);
      changed += 1;
    }
  }
}

console.log(`Removed dark variants from ${changed} files.`);
