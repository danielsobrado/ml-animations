import fs from 'fs';
import path from 'path';

const root = process.cwd();
const designSystem = path.join(root, 'unified-app', 'src', '_design-system');

const fontLinks = `    <!-- Google Fonts: Inter (sans), Source Serif 4 (serif), JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap"
      rel="stylesheet" />
`;

function updatePackage(dir) {
  const file = path.join(dir, 'package.json');
  if (!fs.existsSync(file)) return;

  const pkg = JSON.parse(fs.readFileSync(file, 'utf8'));
  pkg.dependencies = pkg.dependencies || {};
  pkg.dependencies.katex = pkg.dependencies.katex || '^0.16.11';
  fs.writeFileSync(file, `${JSON.stringify(pkg, null, 2)}\n`);
}

function updateHtml(dir) {
  const file = path.join(dir, 'index.html');
  if (!fs.existsSync(file)) return;

  let source = fs.readFileSync(file, 'utf8');
  if (source.includes('Source+Serif+4')) return;

  source = source.replace(/(\s*<title>)/, `\n${fontLinks}$1`);
  fs.writeFileSync(file, source);
}

function updateCss(dir) {
  const srcDir = path.join(dir, 'src');
  if (!fs.existsSync(srcDir)) return;

  const file = path.join(srcDir, 'index.css');
  const importLine = "@import './_design-system/distill.css';";
  let source = fs.existsSync(file)
    ? fs.readFileSync(file, 'utf8')
    : '@tailwind base;\n@tailwind components;\n@tailwind utilities;\n';

  source = source.replace(importLine, '').trimStart();
  fs.writeFileSync(file, `${importLine}\n\n${source}`);
}

function copyDesignSystem(dir) {
  const srcDir = path.join(dir, 'src');
  if (!fs.existsSync(srcDir)) return;
  const target = path.join(srcDir, '_design-system');
  if (path.resolve(designSystem) === path.resolve(target)) return;
  fs.cpSync(designSystem, target, {
    recursive: true,
    force: true,
  });
}

function portStandaloneGradient() {
  const reference = path.join(root, 'unified-app', 'src', 'animations', 'gradient-descent');
  const target = path.join(root, 'gradient-descent-animation', 'src');
  if (!fs.existsSync(reference) || !fs.existsSync(target)) return;

  const files = new Map([
    ['index.jsx', 'App.jsx'],
    ['GradientDescentPanel.jsx', 'GradientDescentPanel.jsx'],
    ['LossHistoryPanel.jsx', 'LossHistoryPanel.jsx'],
    ['PracticePanel.jsx', 'PracticePanel.jsx'],
  ]);

  for (const [fromName, toName] of files) {
    const from = path.join(reference, fromName);
    if (!fs.existsSync(from)) continue;
    const next = fs.readFileSync(from, 'utf8').replaceAll('../../_design-system/ui', './_design-system/ui');
    fs.writeFileSync(path.join(target, toName), next);
  }
}

function cleanCopiedReferenceText() {
  const roots = [
    path.join(root, 'unified-app', 'src', '_design-system'),
    path.join(root, 'unified-app', 'src', 'animations', 'gradient-descent'),
    path.join(root, 'gradient-descent-animation', 'src'),
  ];
  const replacements = new Map([
    ['\u00c2\u00b7', ' / '],
    ['\u00e2\u20ac\u201d', '-'],
    ['\u00e2\u20ac\u00a6', '...'],
    ['\u00e2\u2020\u2019', '->'],
    ['\u00e2\u2013\u00b6', 'Run'],
    ['\u00e2\u201e\u2019', 'L'],
    ['\u00ce\u00b1', 'alpha'],
    ['\u00e2\u2030\u00a5', '>='],
    ['\u00e2\u02c6\u2019', '-'],
  ]);

  const visit = (dir) => {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) visit(file);
      if (!entry.isFile() || !/\.(jsx|css)$/.test(entry.name)) continue;

      let source = fs.readFileSync(file, 'utf8');
      let next = source;
      for (const [bad, good] of replacements) {
        next = next.split(bad).join(good);
      }
      if (next !== source) fs.writeFileSync(file, next);
    }
  };

  roots.forEach(visit);
}

const animationApps = fs.readdirSync(root, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && entry.name.endsWith('-animation'))
  .map((entry) => path.join(root, entry.name))
  .filter((dir) => fs.existsSync(path.join(dir, 'package.json')) && fs.existsSync(path.join(dir, 'src')));

for (const appDir of [path.join(root, 'unified-app'), ...animationApps]) {
  updatePackage(appDir);
  updateHtml(appDir);
  updateCss(appDir);
  copyDesignSystem(appDir);
}

portStandaloneGradient();
copyDesignSystem(path.join(root, 'gradient-descent-animation'));
cleanCopiedReferenceText();
for (const appDir of animationApps) copyDesignSystem(appDir);

console.log(`Updated ${animationApps.length + 1} React apps.`);
