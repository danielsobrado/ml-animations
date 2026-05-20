function normalize(value) {
  return String(value || '').toLowerCase();
}

function makeSearchText(parts) {
  return parts.filter(Boolean).join(' ').toLowerCase();
}

const LESSON_SYMBOLS = {
  softmax: 'σ',
  'gradient-descent': '∇',
  'self-attention': 'QKᵀ',
  'attention-mechanism': 'α',
  'matrix-multiplication': 'AB',
  entropy: 'H',
  'linear-regression': 'ŷ',
  'q-learning': 'Q',
};

const SYMBOL_ALIASES = {
  'τ': 'tau temperature',
  'σ': 'sigma softmax',
  '∇': 'nabla gradient',
  'α': 'alpha attention',
  'QKᵀ': 'q k transpose attention',
};

export function buildCommandPaletteItems(animations, glossaryTerms) {
  const lessonItems = animations.map((animation) => ({
    id: `lesson:${animation.id}`,
    kind: 'lesson',
    name: animation.name,
    label: animation.name,
    symbol: LESSON_SYMBOLS[animation.id] || animation.name.slice(0, 1),
    description: animation.description,
    category: animation.categoryName,
    href: `/animation/${animation.id}`,
    searchText: makeSearchText([
      LESSON_SYMBOLS[animation.id],
      SYMBOL_ALIASES[LESSON_SYMBOLS[animation.id]],
      animation.name,
      animation.description,
      animation.categoryName,
      animation.difficulty,
      ...(animation.learningObjectives || []),
    ]),
  }));

  const glossaryItems = glossaryTerms.map((term) => ({
    id: `glossary:${term.id}`,
    kind: 'glossary',
    name: term.term,
    label: term.term,
    symbol: term.symbol || term.term.slice(0, 1),
    description: term.definition,
    category: term.category,
    href: term.href,
    searchText: makeSearchText([
      term.term,
      term.symbol,
      SYMBOL_ALIASES[term.symbol],
      term.category,
      term.definition,
      term.id,
    ]),
  }));

  return [...lessonItems, ...glossaryItems];
}

export function searchCommandPaletteItems(items, query) {
  const normalizedQuery = normalize(query).trim();
  if (!normalizedQuery) return items;

  return items
    .map((item) => {
      const label = normalize(item.label);
      const text = item.searchText || label;
      let score = 0;

      if (label === normalizedQuery) score += 100;
      if (label.startsWith(normalizedQuery)) score += 50;
      if (text.includes(normalizedQuery)) score += 20;
      for (const token of normalizedQuery.split(/\s+/)) {
        if (token && text.includes(token)) score += 3;
      }

      return { item, score };
    })
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.item.label.localeCompare(b.item.label))
    .map((entry) => entry.item);
}
