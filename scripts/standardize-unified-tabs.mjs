import fs from 'fs';
import path from 'path';

const animationsDir = path.join(process.cwd(), 'unified-app', 'src', 'animations');
const files = fs.readdirSync(animationsDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => path.join(animationsDir, entry.name, 'index.jsx'))
  .filter((file) => fs.existsSync(file));

let changed = 0;

for (const file of files) {
  let source = fs.readFileSync(file, 'utf8');
  if (!source.includes('const tabs') || !source.includes('activeTab') || source.includes('Tabs tabs={tabs}')) {
    continue;
  }

  const next = source.replace(
    /\s*\{\/\* Navigation Tabs \*\/\}\s*<nav[\s\S]*?<\/nav>/,
    '\n            <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />'
  );

  if (next === source) continue;
  source = next;

  if (!source.includes("_design-system/ui")) {
    const importLines = source.match(/^(import[\s\S]*?;\r?\n)(?!import)/);
    if (importLines) {
      source = source.replace(importLines[0], `${importLines[0]}import { Tabs } from '../../_design-system/ui';\n`);
    }
  }

  fs.writeFileSync(file, source);
  changed += 1;
}

console.log(`Standardized ${changed} animation tab entrypoints.`);
