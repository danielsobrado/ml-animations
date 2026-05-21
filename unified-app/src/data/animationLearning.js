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
  'core-ml': '\\hat{f}=\\arg\\min_f \\mathcal{L}_{train}(f)\\quad then\\quad score(\\hat{f},D_{test})',
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
  'train-validation-test-split': 'D=D_{train}\\cup D_{val}\\cup D_{test}',
  'cross-validation': '\\operatorname{CV}_k=\\frac{1}{k}\\sum_{i=1}^{k} score_i',
  overfitting: '\\mathcal{L}_{train}\\downarrow\\quad while\\quad \\mathcal{L}_{val}\\uparrow',
  'logistic-regression': 'p(y=1\\mid x)=\\sigma(w^Tx+b)',
  'classification-metrics': 'F_1=2\\cdot\\frac{precision\\cdot recall}{precision+recall}',
  'roc-pr-curves': 'ROC=(FPR,TPR)\\quad PR=(Recall,Precision)',
  regularization: '\\mathcal{L}_{total}=\\mathcal{L}_{data}+\\lambda\\lVert w\\rVert_2^2',
  'computation-graph-backprop': '\\frac{\\partial L}{\\partial w}=\\frac{\\partial L}{\\partial a}\\frac{\\partial a}{\\partial z}\\frac{\\partial z}{\\partial w}',
  'transformer-token-generation': 'x_{t+1}\\sim \\operatorname{Filter}(\\operatorname{softmax}(z_t/\\tau))',
  'gradient-descent': '\\theta_{t+1}=\\theta_t-\\eta\\nabla\\mathcal{L}(\\theta_t)',
  entropy: 'H(X)=-\\sum_x p(x)\\log p(x)',
  'cross-entropy': 'H(p,q)=-\\sum_x p(x)\\log q(x)',
  'cosine-similarity': '\\cos(\\theta)=\\frac{u\\cdot v}{\\lVert u\\rVert\\lVert v\\rVert}',
  pca: 'X_c^TX_c v_i=\\lambda_i v_i',
  'k-means': '\\min_C\\sum_i \\lVert x_i-c_{a_i}\\rVert^2',
  'conditional-probability': 'P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)}',
  'expected-value-variance': '\\operatorname{Var}(X)=\\mathbb{E}[(X-\\mu)^2]',
  'q-learning': "Q(s,a)\\leftarrow Q(s,a)+\\alpha[r+\\gamma\\max Q(s',a')-Q(s,a)]",
  'bloom-filter': 'p\\approx(1-e^{-kn/m})^k',
  pagerank: 'PR(v)=\\frac{1-d}{N}+d\\sum_{u\\in B_v}\\frac{PR(u)}{L(u)}',
};

function cardSet(def, intuition, equation, example, why, tryIt) {
  return {
    def: { body: def },
    int: { body: intuition },
    eqn: { body: equation },
    ex: { body: example },
    why: { body: why },
    do: { body: tryIt },
  };
}

export const LEARNING_CARD_OVERRIDES = {
  'matrix-multiplication': cardSet(
    'Matrix multiplication solves the problem of composing many weighted sums into one reusable operation.',
    'Read each output cell as one row asking one column how strongly they line up.',
    'The math is a row-column dot product: multiply aligned entries, then add the products.',
    'Manipulate one entry in A or B and predict exactly which output cells can change.',
    'Mistake to avoid: this is not elementwise multiplication; order and shape decide what the product means.',
    'Check understanding by computing one highlighted cell before revealing the animation result.',
  ),
  'linear-regression': cardSet(
    'Linear regression solves the problem of fitting a simple numeric trend from features to a continuous target.',
    'The line is a compromise: it moves to reduce all residuals, not to pass through every point.',
    'The math compares predictions with targets through residuals, then chooses parameters that reduce squared error.',
    'Manipulate slope and intercept, then watch which residuals shrink and which grow.',
    'Mistake to avoid: a lower training error alone does not prove the line will generalize.',
    'Check understanding by predicting whether a slope change raises or lowers total residual error.',
  ),
  'train-validation-test-split': cardSet(
    'Data splitting solves the problem of measuring generalization without letting evaluation data shape the model.',
    'Training is practice, validation is coaching, and the test set is the final exam.',
    'The math separates D into train, validation, and test slices with different jobs.',
    'Manipulate validation and test percentages and watch how much data remains for fitting.',
    'Mistake to avoid: repeated test-set tuning quietly turns the test set into validation data.',
    'Check understanding by identifying which split should guide threshold or hyperparameter choices.',
  ),
  'cross-validation': cardSet(
    'Cross-validation solves the problem of model selection depending too much on one validation split.',
    'Each fold gets one turn as validation while the other folds act as training data.',
    'The math averages validation scores across k rotations, but each pipeline must be refit inside the rotation.',
    'Manipulate the fold count and leakage risk, then compare reported validation score with honest signal.',
    'Mistake to avoid: preprocessing before the fold split lets validation information leak into training.',
    'Check understanding by naming which unit, such as user or time, must stay grouped before random splitting.',
  ),
  'logistic-regression': cardSet(
    'Logistic regression solves binary classification by turning a linear score into a class probability.',
    'A straight boundary creates a logit; sigmoid bends that score onto a 0-to-1 scale.',
    'The math is p = sigmoid(w x + b), followed by a decision threshold.',
    'Manipulate weight, bias, or threshold and predict which points flip labels.',
    'Mistake to avoid: a sigmoid output is not automatically well calibrated just because it is between 0 and 1.',
    'Check understanding by explaining what changes when the threshold moves from 0.5 to 0.7.',
  ),
  'classification-metrics': cardSet(
    'Classification metrics solve the problem of describing which mistakes a classifier is making.',
    'The confusion matrix is an accounting table for positive and negative decisions.',
    'The math builds precision, recall, F1, and accuracy from TP, FP, FN, and TN counts.',
    'Manipulate the threshold and predict which count changes before reading the metrics.',
    'Mistake to avoid: high accuracy can hide poor recall when positives are rare.',
    'Check understanding by choosing a metric for a case where false negatives are expensive.',
  ),
  'roc-pr-curves': cardSet(
    'ROC and PR curves solve the problem of evaluating every threshold instead of trusting one cutoff.',
    'Slide the threshold and the classifier trades found positives against false alarms.',
    'The math plots TPR against FPR for ROC, and precision against recall for PR.',
    'Manipulate the threshold and watch the active point move on both curves.',
    'Mistake to avoid: a high ROC-AUC does not choose a deployment threshold or guarantee strong rare-positive precision.',
    'Check understanding by choosing whether ROC or PR is more useful when positives are rare.',
  ),
  overfitting: cardSet(
    'Overfitting explains why a model can look better on training data while becoming worse on new data.',
    'The model starts learning the pattern, then begins chasing quirks of the sample.',
    'The math shows a widening gap: training error keeps falling while validation error rises.',
    'Manipulate model complexity and find the point where validation error is lowest.',
    'Mistake to avoid: the most flexible model is not automatically the best model.',
    'Check understanding by naming the first visual sign that memorization has started.',
  ),
  regularization: cardSet(
    'Regularization solves the problem of models spending too much complexity on weak evidence.',
    'It is a budget: a parameter can be large only if it earns enough predictive value.',
    'The math adds a penalty term to the data loss, controlled by lambda.',
    'Manipulate lambda and watch coefficients shrink while total loss changes.',
    'Mistake to avoid: stronger regularization is not always better; too much penalty underfits.',
    'Check understanding by explaining why validation data should choose lambda.',
  ),
  'gradient-descent': cardSet(
    'Gradient descent solves the problem of improving parameters when the loss surface gives only local slope information.',
    'Each step feels the uphill direction and walks a scaled distance downhill.',
    'The math updates parameters with theta_next = theta - learning_rate * gradient.',
    'Manipulate the learning rate and compare crawling, converging, and overshooting traces.',
    'Mistake to avoid: the gradient is local, so one step does not guarantee the global minimum.',
    'Check understanding by predicting the next step direction from the slope sign.',
  ),
  relu: cardSet(
    'ReLU solves the activation problem by giving networks a simple nonlinearity that keeps positive signal easy to pass.',
    'Positive inputs go through unchanged; negative inputs are shut off.',
    'The math is f(x)=max(0,x), with slope 1 on the active side and 0 on the blocked side.',
    'Manipulate the input across zero and watch both output and local slope switch.',
    'Mistake to avoid: a blocked ReLU has zero local gradient for that example.',
    'Check understanding by identifying whether a negative pre-activation can send gradient backward.',
  ),
  'computation-graph-backprop': cardSet(
    'Backpropagation solves the problem of assigning blame to each parameter in a nested computation.',
    'The forward graph stores local operations; the backward pass reuses them in reverse.',
    'The math is chain rule multiplication of upstream gradients and local derivatives.',
    'Manipulate x, w, b, target, or learning rate and predict the next loss before updating.',
    'Mistake to avoid: backprop is not a separate learning rule; it is the chain rule on a graph.',
    'Check understanding by explaining why a negative ReLU pre-activation blocks the weight gradient.',
  ),
  tokenization: cardSet(
    'Tokenization solves the problem of turning raw text into units a model can index and embed.',
    'The tokenizer decides what the model sees: characters, words, or reusable subword pieces.',
    'The math is mostly discrete mapping from text spans to token ids before vectors enter the model.',
    'Manipulate the input phrase and compare how character, word, BPE, or WordPiece splits change length.',
    'Mistake to avoid: tokens are not the same as words, especially for rare words and punctuation.',
    'Check understanding by predicting which part of a rare word becomes a shared subword.',
  ),
  embeddings: cardSet(
    'Embeddings solve the problem of representing discrete items as vectors that can be compared and transformed.',
    'Nearby vectors often share learned behavior, but the geometry comes from data and objectives.',
    'The math stores each item as a dense vector in R^d and compares vectors with distance or similarity.',
    'Manipulate one vector or query and observe which neighbors become closest.',
    'Mistake to avoid: embedding distance is learned correlation, not guaranteed semantic truth.',
    'Check understanding by explaining why two similar words can still differ along one direction.',
  ),
  pca: cardSet(
    'PCA solves the problem of compressing numeric data while preserving the largest directions of variation.',
    'Imagine rotating the axes until the first axis catches the longest shadow cast by the point cloud.',
    'The math centers X, forms a covariance matrix, and keeps eigenvectors with the largest eigenvalues.',
    'Manipulate correlation, noise, and component count to see how projection changes reconstruction.',
    'Mistake to avoid: PCA does not use labels, so high variance is not automatically predictive signal.',
    'Check understanding by explaining why the blue PC1 axis changes when the cloud rotates.',
  ),
  'k-means': cardSet(
    'K-means solves the problem of grouping unlabeled points around a chosen number of representative centroids.',
    'Each centroid acts like a magnet; points join the nearest one, then the magnet moves to the group average.',
    'The math minimizes within-cluster squared distance by alternating assignment and centroid update steps.',
    'Manipulate k and iteration count to watch assignments, centroids, and inertia change together.',
    'Mistake to avoid: lower inertia from larger k does not automatically mean a better clustering.',
    'Check understanding by predicting which centroid moves when one point changes cluster.',
  ),
  'attention-mechanism': cardSet(
    'Attention solves the problem of selecting useful context instead of compressing everything equally.',
    'A query asks questions of keys; the answers become weights over values.',
    'The math scores query-key matches, applies softmax, then forms a weighted sum of values.',
    'Manipulate a query or key and predict which value receives the largest weight.',
    'Mistake to avoid: attention weights are contextual mixtures, not permanent word importance scores.',
    'Check understanding by tracing one high score through softmax into the output vector.',
  ),
  'self-attention': cardSet(
    'Self-attention solves the problem of letting every token build context from the same sequence.',
    'Each token creates its own query, then mixes value information from other positions.',
    'The math uses softmax(QK^T / sqrt(d_k))V so dot-product scale does not overwhelm softmax.',
    'Manipulate one token vector and inspect which row of attention weights changes.',
    'Mistake to avoid: self-attention recomputes mixtures for each sequence, not a fixed lookup table.',
    'Check understanding by explaining one row of weights and the context vector it creates.',
  ),
  transformer: cardSet(
    'A transformer solves sequence modeling by stacking attention, feed-forward transforms, residual paths, and normalization.',
    'Attention moves information across positions; feed-forward layers transform each position; residuals preserve the stream.',
    'The math alternates token mixing and per-token nonlinear transforms inside repeated blocks.',
    'Manipulate or step through a token path and watch what attention changes versus what the MLP changes.',
    'Mistake to avoid: a transformer is not only attention; residual, normalization, and feed-forward layers are essential.',
    'Check understanding by naming what information is mixed across tokens and what is processed per token.',
  ),
  'transformer-token-generation': cardSet(
    'Token generation solves the problem of turning a trained transformer into text one next-token decision at a time.',
    'The model reads the current context, scores possible next tokens, chooses one, appends it, and repeats.',
    'The math applies softmax to logits divided by temperature, then filters candidates before selecting a token.',
    'Manipulate temperature, top-k, and top-p to see which tokens remain eligible for the next step.',
    'Mistake to avoid: generation is not a single full-sentence prediction; every sampled token changes the next context.',
    'Check understanding by tracing how one selected token becomes part of the KV cache for the following step.',
  ),
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

  const cards = CARD_TYPES.map((type) => {
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

  if (animation.id === 'softmax') {
    cards.push(
      {
        type: 'theorem',
        label: 'thm.',
        title: 'Theorem 3.1: translation invariance',
        body: 'Translation invariance says adding the same constant to every logit leaves softmax unchanged, because the common exponential factor cancels from numerator and denominator.',
        equation: '\\operatorname{softmax}(z+c\\mathbf{1})=\\operatorname{softmax}(z)',
        terms: termSet(glossary, ['logits', 'temperature', 'probability', 'normalization']),
      },
      {
        type: 'marginalia',
        label: 'note.',
        title: 'Overflow margin note',
        body: 'Notebook aside: implementations subtract the largest logit before exponentiating. The probabilities are identical, but the numbers stay away from overflow.',
        terms: termSet(glossary, ['logits', 'probability', 'normalization', 'temperature']),
      },
    );
  }

  const overrides = LEARNING_CARD_OVERRIDES[animation.id];
  if (!overrides) return cards;

  return cards.map((card) => ({
    ...card,
    ...(overrides[card.type] || {}),
  }));
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
