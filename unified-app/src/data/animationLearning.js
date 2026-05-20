import { getGlossaryTermsForCategory } from './glossaryRepository.js';
import { curriculumTracks } from './animations.js';

export const CARD_TYPES = [
  { id: 'def', label: 'def.', title: 'Definition' },
  { id: 'int', label: 'int.', title: 'Intuition' },
  { id: 'eqn', label: 'eqn.', title: 'Equation' },
  { id: 'ex', label: 'ex.', title: 'Example' },
  { id: 'why', label: 'why.', title: 'Why it matters' },
  { id: 'do', label: 'do.', title: 'Try it' },
];

export const MATH_CONTROLS = [
  { id: 'prereq', sigil: '∂', label: 'Prereq' },
  { id: 'reset', sigil: '↺', label: 'Reset' },
  { id: 'play', sigil: '▷', label: 'Focus' },
  { id: 'sum', sigil: 'Σ', label: 'Glossary' },
  { id: 'next', sigil: '∇', label: 'Next' },
];

const CATEGORY_EQUATIONS = {
  nlp: 'x_{text} \\rightarrow v \\in \\mathbb{R}^d',
  transformers: '\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}(QK^T/\\sqrt{d_k})V',
  'neural-networks': 'h_{\\ell}=\\sigma(W_{\\ell}h_{\\ell-1}+b_{\\ell})',
  'advanced-models': 'p_\\theta(y\\mid x,c)=\\int p_\\theta(y,z\\mid x,c)\\,dz',
  'math-fundamentals': 'f(x;\\theta) \\rightarrow \\arg\\min_\\theta \\mathcal{L}(\\theta)',
  'probability-stats': '\\mathbb{E}[X]=\\sum_x x\\,p(x)',
  'reinforcement-learning': "Q(s,a) \\leftarrow r+\\gamma\\max_{a'}Q(s',a')",
  algorithms: 'score(v)=\\sum_{u\\rightarrow v} w_{uv}\\,score(u)',
  'diffusion-models': 'dx_t=v_\\theta(x_t,t)\\,dt',
};

const EQUATION_OVERRIDES = {
  relu: 'f(x)=\\max(0,x)',
  'leaky-relu': 'f(x)=\\max(\\alpha x,x)',
  softmax: 'p_i=\\frac{e^{z_i}}{\\sum_j e^{z_j}}',
  'matrix-multiplication': 'C_{ij}=\\sum_k A_{ik}B_{kj}',
  eigenvalue: 'Av=\\lambda v',
  svd: 'A=U\\Sigma V^T',
  'qr-decomposition': 'A=QR',
  'linear-regression': '\\hat{y}=X\\beta+\\epsilon',
  'gradient-descent': '\\theta_{t+1}=\\theta_t-\\eta\\nabla\\mathcal{L}(\\theta_t)',
  entropy: 'H(X)=-\\sum_x p(x)\\log p(x)',
  'cross-entropy': 'H(p,q)=-\\sum_x p(x)\\log q(x)',
  'cosine-similarity': '\\cos(\\theta)=\\frac{u\\cdot v}{\\lVert u\\rVert\\lVert v\\rVert}',
  'conditional-probability': 'P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}',
  'expected-value-variance': '\\operatorname{Var}(X)=\\mathbb{E}[(X-\\mu)^2]',
  'q-learning': "Q(s,a)\\leftarrow Q(s,a)+\\alpha[r+\\gamma\\max Q(s',a')-Q(s,a)]",
  'bloom-filter': 'p\\approx(1-e^{-kn/m})^k',
  pagerank: 'PR(v)=\\frac{1-d}{N}+d\\sum_{u\\in B_v}\\frac{PR(u)}{L(u)}',
};

function slugify(value) {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function getCategoryAnimations(animation, allAnimations) {
  return allAnimations.filter((item) => item.categoryId === animation.categoryId);
}

function toNode(animation) {
  return {
    id: animation.id,
    label: animation.name,
    description: animation.description,
  };
}

function getNeighborNodes(animation, allAnimations, direction) {
  const categoryAnimations = getCategoryAnimations(animation, allAnimations);
  const categoryIndex = categoryAnimations.findIndex((item) => item.id === animation.id);
  const globalIndex = allAnimations.findIndex((item) => item.id === animation.id);
  const nodes = [];

  if (direction === 'prev') {
    if (categoryIndex > 0) nodes.push(toNode(categoryAnimations[categoryIndex - 1]));
    if (globalIndex > 0) nodes.push(toNode(allAnimations[globalIndex - 1]));
  } else {
    if (categoryIndex < categoryAnimations.length - 1) nodes.push(toNode(categoryAnimations[categoryIndex + 1]));
    if (globalIndex < allAnimations.length - 1) nodes.push(toNode(allAnimations[globalIndex + 1]));
  }

  const fallback = direction === 'prev'
    ? categoryAnimations.find((item) => item.id !== animation.id) || allAnimations.find((item) => item.id !== animation.id)
    : [...categoryAnimations].reverse().find((item) => item.id !== animation.id) || [...allAnimations].reverse().find((item) => item.id !== animation.id);
  if (nodes.length === 0 && fallback) nodes.push(toNode(fallback));

  return nodes.filter((node, index, list) => (
    node.id !== animation.id && list.findIndex((candidate) => candidate.id === node.id) === index
  ));
}

function byId(allAnimations) {
  return new Map(allAnimations.map((animation) => [animation.id, animation]));
}

function getPrereqNodes(animation, allAnimations) {
  const animationById = byId(allAnimations);
  const nodes = (animation.prerequisites || [])
    .map((id) => animationById.get(id))
    .filter(Boolean)
    .filter((item) => item.id !== animation.id)
    .map(toNode);

  if (nodes.length > 0) return nodes;
  return getNeighborNodes(animation, allAnimations, 'prev').slice(0, 2);
}

function getTrackNextNodes(animation, allAnimations) {
  const animationById = byId(allAnimations);
  const nodes = [];

  for (const trackId of animation.trackIds || []) {
    const track = curriculumTracks.find((candidate) => candidate.id === trackId);
    if (!track) continue;

    const index = track.animationIds.indexOf(animation.id);
    if (index === -1) continue;

    const nextId = track.animationIds[index + 1];
    const nextAnimation = animationById.get(nextId);
    if (nextAnimation && nextAnimation.id !== animation.id) {
      nodes.push(toNode(nextAnimation));
    }
  }

  const uniqueNodes = nodes.filter((node, index, list) => (
    list.findIndex((candidate) => candidate.id === node.id) === index
  ));

  if (uniqueNodes.length > 0) return uniqueNodes.slice(0, 3);
  return getNeighborNodes(animation, allAnimations, 'next').slice(0, 2);
}

function formatPrereqChip(prereqs) {
  if (!prereqs.length) return 'No prerequisites';
  return prereqs.slice(0, 3).map((node) => node.label).join(', ');
}

function termSet(glossary, requestedTerms) {
  const available = new Map(glossary.flatMap((entry) => [
    [entry.id, entry],
    [entry.term.toLowerCase(), entry],
  ]));

  return requestedTerms
    .map((term) => available.get(slugify(term)) || available.get(String(term).toLowerCase()))
    .filter(Boolean)
    .slice(0, 4);
}

function makeCards(animation, glossary, equation) {
  const category = animation.categoryName.toLowerCase();
  const terms = glossary.map((entry) => entry.id);

  return CARD_TYPES.map((type) => {
    const common = {
      type: type.id,
      label: type.label,
      title: type.title,
    };

    if (type.id === 'def') {
      return {
        ...common,
        body: `${animation.name} studies how ${animation.description.toLowerCase()} becomes a repeatable computational concept.`,
        terms: termSet(glossary, [terms[0], 'concept', 'input', 'output']),
      };
    }

    if (type.id === 'int') {
      return {
        ...common,
        body: 'Read the stage as a flow: an input representation is transformed, compared, or updated until the key pattern becomes visible.',
        terms: termSet(glossary, ['input', 'representation', 'vector', terms[1]]),
      };
    }

    if (type.id === 'eqn') {
      return {
        ...common,
        body: 'The headline equation compresses the central operation into symbols so you can connect the visual motion to the math.',
        equation,
        terms: termSet(glossary, ['matrix', 'parameter', 'gradient', terms[2]]),
      };
    }

    if (type.id === 'ex') {
      return {
        ...common,
        body: 'Example lens: change one value, token, state, or vector and watch how the downstream output responds.',
        terms: termSet(glossary, ['token', 'state', 'vector', 'output']),
      };
    }

    if (type.id === 'why') {
      return {
        ...common,
        body: `This matters because ${category} systems only become reliable when you can explain the objective and the failure mode.`,
        terms: termSet(glossary, ['objective', 'loss', 'probability', 'normalization']),
      };
    }

    return {
      ...common,
      body: `${animation.commonMisconception} Do one pass slowly: predict the update, run the animation, then compare your prediction with the output.`,
      terms: termSet(glossary, ['input', 'iteration', 'gradient', 'output']),
    };
  });
}

export function createLearningModel(animation, allAnimations) {
  const glossary = getGlossaryTermsForCategory(animation.categoryId).slice(0, 12);
  const headlineLatex = EQUATION_OVERRIDES[animation.id] || CATEGORY_EQUATIONS[animation.categoryId] || 'y=f(x)';
  const prereqs = getPrereqNodes(animation, allAnimations);
  const next = getTrackNextNodes(animation, allAnimations);

  return {
    conceptName: animation.name,
    headlineEquation: {
      latex: headlineLatex,
    },
    chips: {
      difficulty: animation.difficulty || 'intermediate',
      prereq: formatPrereqChip(prereqs),
      category: animation.categoryName,
      minutes: `${animation.estimatedMinutes || 15} min`,
    },
    mindmap: {
      prereqs,
      current: toNode(animation),
      next,
    },
    learningCards: makeCards(animation, glossary, headlineLatex),
    controls: MATH_CONTROLS,
    glossary,
  };
}
