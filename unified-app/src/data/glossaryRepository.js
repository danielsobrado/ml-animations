function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function makeTermImage(term, category, accent = '#264273') {
  const safeTerm = term.replace(/[<>&"]/g, '');
  const safeCategory = category.replace(/[<>&"]/g, '');
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="360" height="210" viewBox="0 0 360 210" role="img" aria-label="${safeTerm}">
      <rect width="360" height="210" fill="#fbf8f1"/>
      <path d="M28 158 C78 62 125 62 174 158 S276 245 330 98" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      <path d="M28 158 L330 158" stroke="#d9d2c0" stroke-width="1"/>
      <path d="M62 34 L62 180" stroke="#ece6d3" stroke-width="1"/>
      <circle cx="82" cy="96" r="8" fill="${accent}"/>
      <circle cx="180" cy="158" r="8" fill="#a85a3a"/>
      <circle cx="284" cy="104" r="8" fill="#3a6a3a"/>
      <text x="26" y="42" fill="#1a1a1a" font-family="Georgia, serif" font-size="29">${safeTerm}</text>
      <text x="28" y="64" fill="#4a4a4a" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" letter-spacing="2">${safeCategory.toUpperCase()}</text>
      <text x="292" y="184" fill="#b6ac93" font-family="Georgia, serif" font-size="32" font-style="italic">∂Σ</text>
    </svg>
  `.trim();

  return {
    src: `data:image/svg+xml,${encodeURIComponent(svg)}`,
    alt: `${term} glossary visual`,
  };
}

const RAW_TERMS = [
  ['concept', 'Core Math', 'A compact idea that explains how a family of examples behaves.'],
  ['input', 'Core Math', 'The data entering the model, algorithm, or mathematical operation.'],
  ['output', 'Core Math', 'The result produced after applying the current operation.'],
  ['parameter', 'Optimization', 'A tunable value that controls how a model transforms inputs.'],
  ['gradient', 'Optimization', 'A direction and magnitude showing how a quantity changes.'],
  ['loss', 'Optimization', 'A score that measures mismatch between prediction and target.'],
  ['vector', 'Linear Algebra', 'An ordered list of numbers that represents magnitude and direction.'],
  ['matrix', 'Linear Algebra', 'A rectangular table of numbers used to transform vectors.'],
  ['probability', 'Probability', 'A number from 0 to 1 describing how likely an event is.'],
  ['representation', 'Modeling', 'A useful encoding of raw information for computation.'],
  ['normalization', 'Neural Networks', 'A rescaling step that makes values easier to compare or optimize.'],
  ['attention', 'Transformers', 'A weighted lookup that lets one item use information from other items.'],
  ['token', 'NLP', 'A text unit such as a word, subword, or character.'],
  ['embedding', 'NLP', 'A vector representation that places similar items near each other.'],
  ['vocabulary', 'NLP', 'The set of symbols a text model can process directly.'],
  ['query', 'Transformers', 'A vector asking what information is needed.'],
  ['key', 'Transformers', 'A vector advertising what information a position contains.'],
  ['value', 'Transformers', 'The content vector mixed after attention weights are computed.'],
  ['head', 'Transformers', 'One parallel attention channel with its own learned projections.'],
  ['activation', 'Neural Networks', 'A nonlinear function that lets a network model curved relationships.'],
  ['layer', 'Neural Networks', 'A stage of learned computation inside a network.'],
  ['backpropagation', 'Neural Networks', 'The chain-rule procedure for sending gradients backward.'],
  ['basis', 'Linear Algebra', 'A coordinate system for describing vectors.'],
  ['objective', 'Optimization', 'The quantity being optimized or analyzed.'],
  ['derivative', 'Calculus', 'The instantaneous rate of change of a function.'],
  ['random variable', 'Probability', 'A numeric outcome of a random process.'],
  ['distribution', 'Probability', 'A rule assigning probabilities across possible outcomes.'],
  ['expectation', 'Probability', 'The long-run average value of a random variable.'],
  ['state', 'Reinforcement Learning', 'A snapshot of what the agent knows about the environment.'],
  ['action', 'Reinforcement Learning', 'A choice the agent can make.'],
  ['reward', 'Reinforcement Learning', 'Feedback that tells the agent how useful an action was.'],
  ['hash', 'Algorithms', 'A deterministic mapping from data to a compact code.'],
  ['graph', 'Algorithms', 'A set of nodes connected by edges.'],
  ['iteration', 'Algorithms', 'One repeated pass of an update rule.'],
  ['latent', 'Diffusion', 'A compressed hidden representation used for generation.'],
  ['noise', 'Diffusion', 'Random variation added or removed during generation.'],
  ['scheduler', 'Diffusion', 'A rule that chooses the time or noise steps of a process.'],
];

const ACCENTS = {
  'Core Math': '#264273',
  Optimization: '#3a6a3a',
  'Linear Algebra': '#264273',
  Probability: '#3a6a3a',
  Modeling: '#264273',
  'Neural Networks': '#3a6a3a',
  Transformers: '#264273',
  NLP: '#264273',
  Calculus: '#a85a3a',
  'Reinforcement Learning': '#a85a3a',
  Algorithms: '#a85a3a',
  Diffusion: '#264273',
};

export const glossaryTerms = RAW_TERMS.map(([term, category, definition]) => {
  const slug = slugify(term);

  return {
    id: slug,
    slug,
    term,
    category,
    definition,
    href: `/glossary/${slug}`,
    image: makeTermImage(term, category, ACCENTS[category]),
  };
});

const glossaryById = new Map(glossaryTerms.map((term) => [term.id, term]));
const glossaryByTerm = new Map(glossaryTerms.map((term) => [term.term.toLowerCase(), term]));

export const GLOSSARY_IDS_BY_CATEGORY = {
  nlp: ['token', 'embedding', 'vocabulary', 'vector', 'representation', 'input', 'output', 'concept'],
  transformers: ['attention', 'query', 'key', 'value', 'head', 'vector', 'matrix', 'probability', 'normalization', 'representation'],
  'neural-networks': ['activation', 'layer', 'backpropagation', 'gradient', 'loss', 'parameter', 'input', 'output', 'normalization'],
  'advanced-models': ['embedding', 'latent', 'probability', 'representation', 'parameter', 'loss', 'input', 'output'],
  'math-fundamentals': ['matrix', 'vector', 'basis', 'objective', 'derivative', 'gradient', 'iteration', 'concept', 'parameter'],
  'probability-stats': ['probability', 'random-variable', 'distribution', 'expectation', 'loss', 'output', 'concept', 'input'],
  'reinforcement-learning': ['state', 'action', 'reward', 'iteration', 'objective', 'probability', 'input', 'output'],
  algorithms: ['hash', 'graph', 'iteration', 'probability', 'input', 'output', 'concept', 'representation'],
  'diffusion-models': ['latent', 'noise', 'scheduler', 'attention', 'embedding', 'representation', 'input', 'output', 'probability'],
};

export function getGlossaryTerm(idOrTerm) {
  if (!idOrTerm) return undefined;
  const normalized = slugify(idOrTerm);
  return glossaryById.get(normalized) || glossaryByTerm.get(String(idOrTerm).toLowerCase());
}

export function getGlossaryTerms(ids) {
  return ids.map((id) => getGlossaryTerm(id)).filter(Boolean);
}

export function getGlossaryTermsForCategory(categoryId) {
  return getGlossaryTerms(GLOSSARY_IDS_BY_CATEGORY[categoryId] || GLOSSARY_IDS_BY_CATEGORY['math-fundamentals']);
}
