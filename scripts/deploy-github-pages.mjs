import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const appDir = path.join(repoRoot, 'unified-app');
const distDir = path.join(appDir, 'dist');
const nodeModulesDir = path.join(appDir, 'node_modules');
const viteBin = path.join(nodeModulesDir, '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite');
const deployDir = path.join(repoRoot, '.deploy', 'gh-pages');
const branch = process.env.PAGES_BRANCH || 'gh-pages';
const installMode = process.env.PAGES_INSTALL || 'auto';

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

function copyDist() {
  if (!fs.existsSync(distDir)) {
    throw new Error(`Build output not found: ${distDir}`);
  }

  fs.cpSync(distDir, deployDir, { recursive: true });
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
