import { lazy } from 'react';

// Registry of available animation components
// As you integrate more animations, add them here
const animationRegistry = {
  'attention-mechanism': lazy(() => import('./attention-mechanism')),
  // Add more animations as they are integrated:
  // 'transformer': lazy(() => import('./transformer')),
  // 'bert': lazy(() => import('./bert')),
  // 'word2vec': lazy(() => import('./word2vec')),
  // etc.
};

export function getAnimationComponent(id) {
  return animationRegistry[id] || null;
}

export function isAnimationAvailable(id) {
  return id in animationRegistry;
}

export const availableAnimations = Object.keys(animationRegistry);
