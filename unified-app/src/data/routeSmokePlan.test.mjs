import assert from 'node:assert/strict';
import test from 'node:test';

import { toUniqueRoutes, normalizeRoute, EXPLICIT_ROUTES } from '../../scripts/route-smoke-plan.mjs';
import { allAnimations } from './animations.js';

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
  assert.equal(normalizeRoute('/'), '/ml-animations/');
  assert.equal(normalizeRoute('/animation/linear-regression'), '/ml-animations/animation/linear-regression');
  assert.equal(normalizeRoute('/ml-animations/animation/softmax'), '/ml-animations/animation/softmax');
});
