import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const root = process.cwd();
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

const apps = fs.readdirSync(root, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && entry.name.endsWith('-animation'))
  .map((entry) => path.join(root, entry.name))
  .filter((dir) => fs.existsSync(path.join(dir, 'package.json')));

const results = [];

for (const app of apps) {
  const name = path.basename(app);
  const pkg = JSON.parse(fs.readFileSync(path.join(app, 'package.json'), 'utf8'));
  if (!pkg.scripts?.build) {
    results.push({ name, status: 'skipped', reason: 'no build script' });
    continue;
  }

  const result = spawnSync(npm, ['run', 'build', '--silent'], {
    cwd: app,
    encoding: 'utf8',
    shell: process.platform === 'win32',
  });

  if (result.status === 0) {
    results.push({ name, status: 'passed' });
  } else {
    const output = `${result.error?.message || ''}\n${result.stdout || ''}\n${result.stderr || ''}`
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-18)
      .join('\n');
    results.push({ name, status: 'failed', code: result.status, output });
  }

  console.log(`${name}: ${result.status === 0 ? 'passed' : 'failed'}`);
}

const passed = results.filter((item) => item.status === 'passed').length;
const failed = results.filter((item) => item.status === 'failed');
const skipped = results.filter((item) => item.status === 'skipped').length;

console.log('\nStandalone build sweep');
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed.length}`);
console.log(`Skipped: ${skipped}`);

for (const item of failed) {
  console.log(`\n[${item.name}]`);
  console.log(item.output);
}

process.exit(failed.length ? 1 : 0);
