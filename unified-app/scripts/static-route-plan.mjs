import { allAnimations } from '../src/data/animations.js';
import { glossaryTerms } from '../src/data/glossaryRepository.js';

export function toStaticRouteDirectories(animations = allAnimations, terms = glossaryTerms) {
  const animationRoutes = animations.map((animation) => ['animation', animation.id]);
  const glossaryRoutes = terms.map((term) => ['glossary', term.slug]);

  return [['labs'], ['settings'], ['glossary'], ...animationRoutes, ...glossaryRoutes];
}
