import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { allAnimations } from '../unified-app/src/data/animations.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const siteBaseUrl = 'https://danielsobrado.github.io/ml-animations';
const appBasePath = '/ml-animations';

const escapeHtml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const renderPage = (animation) => {
  const title = escapeHtml(`${animation.name} | Machine Learning Visualized`);
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
    <meta name="keywords" content="${escapeHtml(`${animation.name}, machine learning visualization, Machine Learning Visualized`)}" />
    <meta name="robots" content="index, follow" />
    <meta http-equiv="refresh" content="0; url=${route}" />
    <meta property="og:type" content="website" />
    <meta property="og:title" content="${title}" />
    <meta property="og:description" content="${description}" />
    <meta property="og:url" content="${canonical}" />
    <meta property="og:site_name" content="Machine Learning Visualized" />
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
};

let created = 0;

for (const animation of allAnimations) {
  const dir = path.join(repoRoot, `${animation.id}-animation`);
  const file = path.join(dir, 'index.html');
  if (fs.existsSync(file)) continue;

  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(file, renderPage(animation));
  created += 1;
}

console.log(`Created ${created} static animation entry page${created === 1 ? '' : 's'}.`);
