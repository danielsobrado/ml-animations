import { spawn } from 'node:child_process';
import { execFile } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../../..');
const appRoot = path.join(repoRoot, 'unified-app');
const appRequire = createRequire(path.join(appRoot, 'package.json'));
const { chromium } = appRequire('playwright');
const screenshotRoot = path.join(repoRoot, 'screenshots', 'theme-audit');
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const outDir = path.join(screenshotRoot, timestamp);
const port = Number(process.env.THEME_AUDIT_PORT || 5199);
const serverUrl = `http://127.0.0.1:${port}`;
const baseUrl = `${serverUrl}/ml-animations`;
const channel = process.env.THEME_AUDIT_CHANNEL || 'chrome';

const args = new Set(process.argv.slice(2));
const quick = args.has('--quick');
const keepServer = args.has('--keep-server');
const routeFilter = valueAfter('--route');
const limit = Number(valueAfter('--limit') || 0);

function valueAfter(name) {
  const idx = process.argv.indexOf(name);
  return idx >= 0 ? process.argv[idx + 1] : '';
}

function slug(input) {
  return String(input)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'screen';
}

async function readCatalog() {
  const source = await readFile(path.join(appRoot, 'src', 'data', 'animations.js'), 'utf8');
  const matches = [...source.matchAll(/\{\s*id:\s*'([^']+)',\s*name:\s*'([^']+)',\s*icon:\s*[^,]+,\s*description:\s*'([^']*)'/g)];
  let routes = matches.map((match) => ({
    id: match[1],
    name: match[2],
    description: match[3],
    url: `${baseUrl}/animation/${match[1]}`,
  }));

  if (routeFilter) {
    routes = routes.filter((route) => route.id.includes(routeFilter));
  }
  if (quick) {
    routes = routes.slice(0, 8);
  }
  if (limit > 0) {
    routes = routes.slice(0, limit);
  }
  return routes;
}

async function waitForServer(url, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Keep polling until Vite is ready.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function startServer() {
  const child = spawn('npm', ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: appRoot,
    shell: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, BROWSER: 'none' },
  });

  child.stdout.on('data', (data) => process.stdout.write(`[vite] ${data}`));
  child.stderr.on('data', (data) => process.stderr.write(`[vite] ${data}`));
  return child;
}

async function stopServer(child) {
  if (!child?.pid) return;
  if (process.platform === 'win32') {
    await new Promise((resolve) => {
      execFile('taskkill', ['/pid', String(child.pid), '/t', '/f'], () => resolve());
    });
    return;
  }
  child.kill('SIGTERM');
}

async function waitForPage(page) {
  await page.waitForLoadState('domcontentloaded');
  await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
  await page.locator('#root').waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(350);
}

async function capture(page, filePath) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await page.screenshot({ path: filePath, fullPage: true });
}

async function themeFindings(page) {
  return page.evaluate(() => {
    const allowedColor = new Set([
      'rgb(251, 248, 241)',
      'rgb(244, 239, 226)',
      'rgb(254, 252, 247)',
      'rgb(26, 26, 26)',
      'rgb(42, 42, 42)',
      'rgb(74, 74, 74)',
      'rgb(122, 122, 120)',
      'rgb(217, 210, 192)',
      'rgb(182, 172, 147)',
      'rgb(236, 230, 211)',
      'rgb(38, 66, 115)',
      'rgb(58, 90, 150)',
      'rgb(168, 90, 58)',
      'rgb(58, 106, 58)',
      'rgb(138, 29, 29)',
      'rgba(0, 0, 0, 0)',
    ]);

    const interesting = [...document.querySelectorAll(
      '.ua-animation-page button, .ua-animation-page nav, .ua-animation-page [class*="bg-"], .ua-animation-page [class*="rounded"], .ua-animation-page [class*="shadow"], .ua-animation-page [class*="text-"], .ua-animation-page [class*="border-"]'
    )];

    const findings = [];
    for (const el of interesting) {
      const style = getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      if (rect.width < 8 || rect.height < 8) continue;

      const label = (el.innerText || el.getAttribute('aria-label') || el.className || el.tagName)
        .toString()
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 100);

      if (style.backgroundImage && style.backgroundImage !== 'none') {
        findings.push({ type: 'gradient-background', label, className: el.className.toString().slice(0, 160) });
      }

      const radius = Math.max(
        parseFloat(style.borderTopLeftRadius) || 0,
        parseFloat(style.borderTopRightRadius) || 0,
        parseFloat(style.borderBottomRightRadius) || 0,
        parseFloat(style.borderBottomLeftRadius) || 0
      );
      if (radius > 8) {
        findings.push({ type: 'large-radius', radius, label, className: el.className.toString().slice(0, 160) });
      }

      if (el.tagName === 'BUTTON' && !allowedColor.has(style.backgroundColor)) {
        findings.push({ type: 'button-background-token', backgroundColor: style.backgroundColor, label, className: el.className.toString().slice(0, 160) });
      }
    }

    return findings.slice(0, 30);
  });
}

async function captureRoute(page, route, manifest) {
  const routeDir = path.join(outDir, 'animations', route.id);
  await page.goto(route.url);
  await waitForPage(page);

  const defaultFile = path.join(routeDir, '00-default.png');
  await capture(page, defaultFile);
  manifest.screens.push({
    route: route.id,
    title: route.name,
    state: 'default',
    file: path.relative(repoRoot, defaultFile).replaceAll('\\', '/'),
    findings: await themeFindings(page),
  });

  const tabLocator = page.locator('.ua-animation-page > div > nav button, .ua-animation-page .ds-tabs button');
  const count = await tabLocator.count();
  for (let index = 0; index < count; index += 1) {
    const label = await tabLocator.nth(index).innerText().catch(() => `tab-${index + 1}`);
    await tabLocator.nth(index).click({ timeout: 5000 }).catch(() => {});
    await waitForPage(page);
    const file = path.join(routeDir, `${String(index + 1).padStart(2, '0')}-${slug(label)}.png`);
    await capture(page, file);
    manifest.screens.push({
      route: route.id,
      title: route.name,
      state: label.replace(/\s+/g, ' ').trim(),
      file: path.relative(repoRoot, file).replaceAll('\\', '/'),
      findings: await themeFindings(page),
    });
  }
}

async function main() {
  if (!existsSync(path.join(appRoot, 'node_modules', 'playwright'))) {
    throw new Error('Playwright is not installed in unified-app. Run: rtk npm install --prefix unified-app');
  }

  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });
  const server = startServer();
  const manifest = {
    generatedAt: new Date().toISOString(),
    baseUrl,
    screens: [],
  };

  try {
    await waitForServer(`${baseUrl}/`);
    const browser = await chromium.launch(channel === 'chromium' ? {} : { channel });
    const page = await browser.newPage({ viewport: { width: 1365, height: 900 }, deviceScaleFactor: 1 });

    await page.goto(`${baseUrl}/`);
    await waitForPage(page);
    const homeFile = path.join(outDir, 'home', 'desktop.png');
    await capture(page, homeFile);
    manifest.screens.push({
      route: 'home',
      state: 'desktop',
      file: path.relative(repoRoot, homeFile).replaceAll('\\', '/'),
      findings: await themeFindings(page),
    });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`${baseUrl}/`);
    await waitForPage(page);
    const mobileFile = path.join(outDir, 'home', 'mobile.png');
    await capture(page, mobileFile);
    manifest.screens.push({
      route: 'home',
      state: 'mobile',
      file: path.relative(repoRoot, mobileFile).replaceAll('\\', '/'),
      findings: await themeFindings(page),
    });

    await page.setViewportSize({ width: 1365, height: 900 });
    const routes = await readCatalog();
    for (const route of routes) {
      console.log(`Capturing ${route.id}`);
      await captureRoute(page, route, manifest);
    }

    await browser.close();
  } finally {
    await writeFile(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
    if (!keepServer) {
      await stopServer(server);
    }
  }

  const findingCount = manifest.screens.reduce((sum, screen) => sum + screen.findings.length, 0);
  console.log(`Captured ${manifest.screens.length} screens in ${path.relative(repoRoot, outDir)}`);
  console.log(`Automated theme findings: ${findingCount}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
