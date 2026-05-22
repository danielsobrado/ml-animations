import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { allAnimations } from '../unified-app/src/data/animations.js';
import { toStaticRouteFiles } from '../unified-app/scripts/static-route-plan.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const appDir = path.join(repoRoot, 'unified-app');
const distDir = path.join(appDir, 'dist');
const nodeModulesDir = path.join(appDir, 'node_modules');
const viteBin = path.join(nodeModulesDir, '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite');
const deployDir = path.join(repoRoot, '.deploy', 'gh-pages');
const branch = process.env.PAGES_BRANCH || 'gh-pages';
const installMode = process.env.PAGES_INSTALL || 'auto';
const siteBaseUrl = 'https://danielsobrado.github.io/ml-animations';
const appBasePath = '/ml-animations';

function run(command, args, options = {}) {
  const useShell = process.platform === 'win32' && ['npm', 'npx'].includes(command);
  const executable = command;
  const result = spawnSync(executable, args, {
    cwd: options.cwd || repoRoot,
    stdio: options.capture ? 'pipe' : 'inherit',
    shell: useShell,
    encoding: 'utf8',
  });

  if (options.allowFailure) {
    return result;
  }

  if (result.error || result.status !== 0) {
    const shown = [command, ...args].join(' ');
    if (result.error) {
      throw new Error(`Command failed: ${shown}\n${result.error.message}`);
    }
    throw new Error(`Command failed: ${shown}`);
  }

  return result;
}

function ensureManagedDeployPath() {
  const relative = path.relative(repoRoot, deployDir);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Refusing to deploy outside the repository: ${deployDir}`);
  }
  if (!relative.startsWith(`.deploy${path.sep}`)) {
    throw new Error(`Refusing to clear unmanaged path: ${deployDir}`);
  }
}

function removeWorktreeIfPresent() {
  if (!fs.existsSync(deployDir)) {
    return;
  }

  run('git', ['worktree', 'remove', '--force', deployDir], { allowFailure: true });
  fs.rmSync(deployDir, { recursive: true, force: true });
}

function emptyDirectoryExceptGit(directory) {
  for (const entry of fs.readdirSync(directory)) {
    if (entry === '.git') {
      continue;
    }
    fs.rmSync(path.join(directory, entry), { recursive: true, force: true });
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderStaticAnimationEntry(animation) {
  const title = escapeHtml(`${animation.name} | ML Animations`);
  const description = escapeHtml(
    animation.description ||
      `Explore the ${animation.name} guided machine-learning lesson with visual intuition and practice checks.`,
  );
  const route = `${appBasePath}/animation/${animation.id}`;
  const canonical = `${siteBaseUrl}/${animation.id}-animation/`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${title}</title>
    <meta name="description" content="${description}" />
    <meta name="keywords" content="${escapeHtml(`${animation.name}, machine learning animation, ML Animations`)}" />
    <meta name="robots" content="index, follow" />
    <meta http-equiv="refresh" content="0; url=${route}" />
    <meta property="og:type" content="website" />
    <meta property="og:title" content="${title}" />
    <meta property="og:description" content="${description}" />
    <meta property="og:url" content="${canonical}" />
    <meta property="og:site_name" content="ML Animations" />
    <meta property="og:image" content="${siteBaseUrl}/favicon.svg" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${title}" />
    <meta name="twitter:description" content="${description}" />
    <meta name="twitter:image" content="${siteBaseUrl}/favicon.svg" />
    <link rel="canonical" href="${canonical}" />
    <link rel="icon" type="image/svg+xml" href="${appBasePath}/favicon.svg" />
</head>
<body>
    <main>
        <h1>${escapeHtml(animation.name)}</h1>
        <p>${description}</p>
        <p><a href="${route}">Open ${escapeHtml(animation.name)}</a></p>
    </main>
    <script>
        window.location.replace('${route}');
    </script>
</body>
</html>
`;
}

function writeStaticAnimationEntryPages() {
  for (const animation of allAnimations) {
    const directory = path.join(deployDir, `${animation.id}-animation`);
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'index.html'), renderStaticAnimationEntry(animation));
  }
}

function writeStaticSpaRoutePages() {
  const indexFile = path.join(deployDir, 'index.html');
  if (!fs.existsSync(indexFile)) {
    throw new Error(`Cannot materialize SPA routes without ${indexFile}`);
  }

  for (const routeParts of toStaticRouteFiles()) {
    const routeFile = path.join(deployDir, ...routeParts);
    fs.mkdirSync(path.dirname(routeFile), { recursive: true });
    fs.copyFileSync(indexFile, routeFile);
  }
}

function copyDist() {
  if (!fs.existsSync(distDir)) {
    throw new Error(`Build output not found: ${distDir}`);
  }

  fs.cpSync(distDir, deployDir, { recursive: true });
  writeStaticSpaRoutePages();
  writeStaticAnimationEntryPages();
  fs.writeFileSync(path.join(deployDir, '.nojekyll'), '');

  const indexFile = path.join(deployDir, 'index.html');
  if (fs.existsSync(indexFile)) {
    fs.copyFileSync(indexFile, path.join(deployDir, '404.html'));
  }
}

function remoteBranchExists() {
  const result = run('git', ['ls-remote', '--exit-code', '--heads', 'origin', branch], {
    capture: true,
    allowFailure: true,
  });
  return result.status === 0;
}

ensureManagedDeployPath();

if (installMode === 'skip' || (installMode === 'auto' && fs.existsSync(viteBin))) {
  console.log('Using existing unified-app dependencies...');
} else {
  console.log('Installing unified-app dependencies...');
  run('npm', ['install', '--legacy-peer-deps'], { cwd: appDir });
}

console.log('Building unified-app...');
run('npm', ['run', 'build'], { cwd: appDir });

console.log(`Preparing ${branch} worktree...`);
fs.mkdirSync(path.dirname(deployDir), { recursive: true });
removeWorktreeIfPresent();
run('git', ['fetch', 'origin', branch], { allowFailure: true });

if (remoteBranchExists()) {
  run('git', ['worktree', 'add', '--force', '-B', branch, deployDir, `origin/${branch}`]);
} else {
  run('git', ['worktree', 'add', '--force', '--detach', deployDir, 'HEAD']);
  run('git', ['checkout', '--orphan', branch], { cwd: deployDir });
}

emptyDirectoryExceptGit(deployDir);
copyDist();

run('git', ['add', '-A'], { cwd: deployDir });
const status = run('git', ['status', '--porcelain'], { cwd: deployDir, capture: true });

if (!status.stdout.trim()) {
  console.log(`No deploy changes to publish on ${branch}.`);
} else {
  run('git', ['commit', '-m', 'Deploy GitHub Pages'], { cwd: deployDir });
  run('git', ['push', '-u', 'origin', branch], { cwd: deployDir });
}

run('git', ['worktree', 'remove', '--force', deployDir], { allowFailure: true });
fs.rmSync(deployDir, { recursive: true, force: true });

console.log(`Published unified-app/dist to ${branch}.`);
