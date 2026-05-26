import assert from 'node:assert/strict';
import test from 'node:test';

import { toUniqueRoutes, normalizeRoute, EXPLICIT_ROUTES } from '../../scripts/route-smoke-plan.mjs';
import { toStaticRouteDirectories } from '../../scripts/static-route-plan.mjs';
import { allAnimations } from './animations.js';
import { glossaryTerms } from './glossaryRepository.js';

test('route smoke plan includes every catalog animation route exactly once', () => {
  const routes = toUniqueRoutes();
  const routeSet = new Set(routes);

  assert.equal(routes.length, routeSet.size, 'smoke routes should be unique');
  for (const animation of allAnimations) {
    assert.ok(routeSet.has(`/animation/${animation.id}`), `${animation.id} should have a smoke route`);
  }
});

test('route smoke plan preserves critical explicit routes', () => {
  const routes = new Set(toUniqueRoutes());

  for (const route of EXPLICIT_ROUTES) {
    assert.ok(routes.has(route), `${route} should be included in route smoke coverage`);
  }
});

test('route smoke paths are normalized under the GitHub Pages base path', () => {
  assert.equal(normalizeRoute('/'), '/Machine-Learning-Visualized/');
  assert.equal(normalizeRoute('/labs'), '/Machine-Learning-Visualized/labs/');
  assert.equal(normalizeRoute('/settings'), '/Machine-Learning-Visualized/settings/');
  assert.equal(normalizeRoute('/animation/linear-regression'), '/Machine-Learning-Visualized/animation/linear-regression/');
  assert.equal(normalizeRoute('/Machine-Learning-Visualized/animation/softmax'), '/Machine-Learning-Visualized/animation/softmax/');
});

test('GitHub Pages static route plan materializes every SPA detail route', () => {
  const staticRoutes = new Set(toStaticRouteDirectories().map((routeParts) => routeParts.join('/')));

  assert.ok(staticRoutes.has('labs'), 'labs should have a static SPA route');
  assert.ok(staticRoutes.has('settings'), 'settings should have a static SPA route');

  for (const animation of allAnimations) {
    assert.ok(staticRoutes.has(`animation/${animation.id}`), `${animation.id} should have a static SPA route`);
  }

  for (const term of glossaryTerms) {
    assert.ok(staticRoutes.has(`glossary/${term.slug}`), `${term.slug} should have a static glossary route`);
  }
});
