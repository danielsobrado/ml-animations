function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function escapeXml(value) {
  return String(value).replace(/[<>&"]/g, (char) => ({
    '<': '&lt;',
    '>': '&gt;',
    '&': '&amp;',
    '"': '&quot;',
  })[char]);
}

function wrapText(text, maxLength = 28) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let current = '';

  words.forEach((word) => {
    const next = current ? `${current} ${word}` : word;
    if (next.length > maxLength && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  });

  if (current) lines.push(current);
  return lines.slice(0, 3);
}

function textBlock(lines, x, y, options = {}) {
  const {
    size = 14,
    fill = '#1f1f1f',
    family = 'Georgia, serif',
    weight = '400',
    leading = 20,
  } = options;

  return lines
    .map((line, index) => (
      `<text x="${x}" y="${y + index * leading}" fill="${fill}" font-family="${family}" font-size="${size}" font-weight="${weight}">${escapeXml(line)}</text>`
    ))
    .join('');
}

function visualForTerm(term, category, accent, visual = 'concept') {
  const muted = '#d8cfba';
  const ink = '#1f1f1f';
  const amber = '#a85a3a';
  const green = '#3f6f46';
  const blue = '#234b8f';
  const title = textBlock(wrapText(term, 18), 24, 38, { size: 25, weight: '500', leading: 27 });
  const categoryLabel = `<text x="26" y="92" fill="#5f574a" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10" letter-spacing="2">${escapeXml(category.toUpperCase())}</text>`;

  const visuals = {
    axis: `
      <path d="M42 164 H318 M68 184 V104" stroke="${muted}" stroke-width="2"/>
      <path d="M52 154 C98 78 142 78 186 154 S272 224 310 108" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      <circle cx="96" cy="105" r="8" fill="${blue}"/>
      <circle cx="188" cy="154" r="8" fill="${amber}"/>
      <circle cx="282" cy="112" r="8" fill="${green}"/>
    `,
    vector: `
      <path d="M68 164 H300 M80 178 V72" stroke="${muted}" stroke-width="2"/>
      <path d="M92 152 L250 82" stroke="${accent}" stroke-width="6" stroke-linecap="round"/>
      <path d="M250 82 L232 84 M250 82 L238 98" stroke="${accent}" stroke-width="6" stroke-linecap="round"/>
      <text x="108" y="128" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="16">[2.1, -0.4, 7.0]</text>
    `,
    matrix: `
      <rect x="82" y="98" width="156" height="82" fill="#fffdf7" stroke="${muted}"/>
      <path d="M134 98 V180 M186 98 V180 M82 125 H238 M82 153 H238" stroke="${muted}"/>
      <text x="96" y="119" fill="${accent}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="15">0.8</text>
      <text x="149" y="147" fill="${amber}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="15">-1</text>
      <text x="198" y="174" fill="${green}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="15">2.4</text>
      <path d="M258 138 L304 138" stroke="${accent}" stroke-width="4"/>
      <path d="M304 138 L290 130 M304 138 L290 146" stroke="${accent}" stroke-width="4"/>
    `,
    probability: `
      <rect x="54" y="146" width="246" height="18" fill="#efe8d8" stroke="${muted}"/>
      <rect x="54" y="146" width="158" height="18" fill="${accent}" opacity="0.85"/>
      <text x="52" y="134" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="15">0</text>
      <text x="292" y="134" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="15">1</text>
      <circle cx="212" cy="155" r="9" fill="${amber}"/>
      <text x="124" y="190" fill="${ink}" font-family="Georgia, serif" font-size="17">likelihood as a share of possibility</text>
    `,
    loss: `
      <path d="M54 166 H312" stroke="${muted}" stroke-width="2"/>
      <path d="M64 126 L118 102 L176 144 L230 116 L292 158" fill="none" stroke="${accent}" stroke-width="5" stroke-linecap="round"/>
      <path d="M64 148 L118 118 L176 118 L230 126 L292 132" fill="none" stroke="${amber}" stroke-width="4" stroke-dasharray="7 7"/>
      <path d="M176 118 V144" stroke="${ink}" stroke-width="2"/>
      <text x="182" y="135" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">error</text>
    `,
    gradient: `
      <path d="M52 166 H314 M78 180 V72" stroke="${muted}" stroke-width="2"/>
      <path d="M70 158 C112 64 188 64 256 158" fill="none" stroke="${accent}" stroke-width="5"/>
      <path d="M178 102 L132 132" stroke="${amber}" stroke-width="6" stroke-linecap="round"/>
      <path d="M132 132 L148 127 M132 132 L142 118" stroke="${amber}" stroke-width="6" stroke-linecap="round"/>
      <text x="200" y="104" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="16">slope</text>
    `,
    attention: `
      <circle cx="92" cy="144" r="20" fill="#fffdf7" stroke="${accent}" stroke-width="3"/>
      <circle cx="178" cy="104" r="20" fill="#fffdf7" stroke="${amber}" stroke-width="3"/>
      <circle cx="258" cy="150" r="20" fill="#fffdf7" stroke="${green}" stroke-width="3"/>
      <path d="M112 138 C142 112 148 106 158 104" stroke="${accent}" stroke-width="5" fill="none"/>
      <path d="M112 150 C160 176 214 174 238 156" stroke="${accent}" stroke-width="2" fill="none"/>
      <text x="82" y="149" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">Q</text>
      <text x="172" y="109" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">K</text>
      <text x="252" y="155" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">V</text>
    `,
    token: `
      <rect x="52" y="116" width="62" height="34" fill="#fffdf7" stroke="${accent}" stroke-width="2"/>
      <rect x="124" y="116" width="82" height="34" fill="#fffdf7" stroke="${amber}" stroke-width="2"/>
      <rect x="216" y="116" width="82" height="34" fill="#fffdf7" stroke="${green}" stroke-width="2"/>
      <text x="67" y="138" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">the</text>
      <text x="138" y="138" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">model</text>
      <text x="232" y="138" fill="${ink}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13">works</text>
      <path d="M82 166 V178 M165 166 V178 M257 166 V178" stroke="${muted}" stroke-width="2"/>
    `,
    graph: `
      <circle cx="82" cy="150" r="14" fill="#fffdf7" stroke="${accent}" stroke-width="3"/>
      <circle cx="166" cy="104" r="14" fill="#fffdf7" stroke="${amber}" stroke-width="3"/>
      <circle cx="252" cy="150" r="14" fill="#fffdf7" stroke="${green}" stroke-width="3"/>
      <path d="M96 143 L152 112 M180 112 L238 143 M98 152 H238" stroke="${muted}" stroke-width="3"/>
      <path d="M238 143 L225 142 M238 143 L230 132" stroke="${muted}" stroke-width="3"/>
    `,
    confusion: `
      <rect x="78" y="100" width="156" height="84" fill="#fffdf7" stroke="${muted}"/>
      <path d="M156 100 V184 M78 142 H234" stroke="${muted}"/>
      <rect x="78" y="100" width="78" height="42" fill="${accent}" opacity="0.18"/>
      <rect x="156" y="142" width="78" height="42" fill="${accent}" opacity="0.18"/>
      <text x="102" y="128" fill="${accent}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18">TP</text>
      <text x="184" y="128" fill="${amber}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18">FP</text>
      <text x="102" y="170" fill="${amber}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18">FN</text>
      <text x="184" y="170" fill="${green}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18">TN</text>
    `,
    diffusion: `
      <rect x="56" y="118" width="46" height="46" fill="#f0ece0" stroke="${muted}"/>
      <rect x="126" y="118" width="46" height="46" fill="#fffdf7" stroke="${accent}"/>
      <rect x="196" y="118" width="46" height="46" fill="#fffdf7" stroke="${amber}"/>
      <rect x="266" y="118" width="46" height="46" fill="#fffdf7" stroke="${green}"/>
      <path d="M106 141 H122 M176 141 H192 M246 141 H262" stroke="${accent}" stroke-width="3"/>
      <circle cx="68" cy="130" r="3" fill="${accent}"/><circle cx="86" cy="152" r="3" fill="${amber}"/><circle cx="94" cy="134" r="3" fill="${green}"/>
      <path d="M136 154 C150 128 158 130 166 122" stroke="${accent}" stroke-width="3" fill="none"/>
      <path d="M206 152 C216 134 226 132 236 124" stroke="${amber}" stroke-width="3" fill="none"/>
      <path d="M276 148 C288 132 298 130 306 124" stroke="${green}" stroke-width="3" fill="none"/>
    `,
  };

  return `
    ${title}
    ${categoryLabel}
    ${visuals[visual] || visuals.axis}
  `;
}

function makeTermImage(term, category, accent = '#264273', visual) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="360" height="210" viewBox="0 0 360 210" role="img" aria-label="${escapeXml(term)}">
      <rect width="360" height="210" fill="#fbf8f1"/>
      <rect x="12" y="12" width="336" height="186" fill="none" stroke="#d8cfba"/>
      ${visualForTerm(term, category, accent, visual)}
      <text x="292" y="184" fill="#b6ac93" font-family="Georgia, serif" font-size="28" font-style="italic">dS</text>
    </svg>
  `.trim();

  return {
    src: `data:image/svg+xml,${encodeURIComponent(svg)}`,
    alt: `${term} glossary visual showing ${visual || 'concept'} structure`,
  };
}

const ACCENTS = {
  'Core Math': '#264273',
  Optimization: '#3a6a3a',
  'Linear Algebra': '#264273',
  Probability: '#3a6a3a',
  Modeling: '#264273',
  'Neural Networks': '#3a6a3a',
  'Core ML': '#1f6f8b',
  Transformers: '#264273',
  NLP: '#264273',
  Calculus: '#a85a3a',
  'Reinforcement Learning': '#a85a3a',
  Algorithms: '#a85a3a',
  Diffusion: '#264273',
};

const SYMBOLS = {
  logits: 'z',
  temperature: 'Ï„',
  attention: 'alpha',
  probability: 'p',
  gradient: 'grad',
  matrix: 'A',
  vector: 'v',
};

const TERM_DETAILS = [
  {
    term: 'concept',
    category: 'Core Math',
    visual: 'axis',
    definition: 'A compact idea that explains how a family of examples behaves.',
    explanation: 'A concept is the reusable pattern behind many individual examples. In this app, a concept is usually the thing you want to recognize even when the numbers, diagrams, or model architecture change.',
    intuition: 'Think of it as the handle you can grab. Once you know the handle, new examples feel less like isolated facts.',
    example: 'The concept of a gradient is not one specific arrow; it is the idea that a function has a local direction of steepest increase.',
    pitfall: 'Do not confuse a concept with a memorized example. The example should serve the idea, not replace it.',
  },
  {
    term: 'input',
    category: 'Core Math',
    visual: 'token',
    definition: 'The data entering the model, algorithm, or mathematical operation.',
    explanation: 'An input is what the current system receives before it does any work. Inputs can be numbers, vectors, images, tokens, states, or previous layer activations depending on the lesson.',
    intuition: 'The input is the question being handed to the machine. The form of the question controls what kinds of answers are possible.',
    example: 'For a classifier, the input might be a row of features; for a transformer, it might be token embeddings.',
    pitfall: 'Raw data is not always the direct input. Preprocessing often turns raw text, pixels, or tables into model-ready values first.',
  },
  {
    term: 'output',
    category: 'Core Math',
    visual: 'axis',
    definition: 'The result produced after applying the current operation.',
    explanation: 'An output is the value returned by a function, layer, model, or algorithm step. It may be a final prediction or an intermediate value passed to the next computation.',
    intuition: 'The output is the answer at the boundary you are inspecting. Move the boundary and the same value may become the next step input.',
    example: 'A softmax layer outputs probabilities; a convolution layer outputs feature maps.',
    pitfall: 'A model output is not automatically a decision. Scores often need thresholds, decoding, ranking, or calibration.',
  },
  {
    term: 'parameter',
    category: 'Optimization',
    visual: 'matrix',
    definition: 'A tunable value that controls how a model transforms inputs.',
    explanation: 'Parameters are learned numbers such as weights, biases, embeddings, or projection matrices. Training changes parameters so the model produces better outputs on the task.',
    intuition: 'Parameters are the knobs the optimizer is allowed to turn. The architecture says where the knobs are; data decides how they should be set.',
    example: 'In linear regression, the slope and intercept are parameters. In a transformer, attention projection matrices are parameters.',
    pitfall: 'Parameters are not the same as hyperparameters. Learning rate and batch size usually guide training but are not learned from each example.',
  },
  {
    term: 'gradient',
    category: 'Optimization',
    visual: 'gradient',
    definition: 'A direction and magnitude showing how a quantity changes.',
    explanation: 'A gradient tells how sensitive an objective is to small changes in each parameter. Optimizers use gradients to decide which way to move parameters during training.',
    intuition: 'A gradient is a slope map. It says which direction is uphill; minimizing loss usually means stepping the other way.',
    example: 'If increasing a weight raises the loss, the gradient for that weight is positive and gradient descent will tend to reduce it.',
    pitfall: 'A gradient is local. It tells what helps near the current point, not a guaranteed route to the global best solution.',
  },
  {
    term: 'loss',
    category: 'Optimization',
    visual: 'loss',
    definition: 'A score that measures mismatch between prediction and target.',
    explanation: 'Loss turns model performance into a number the optimizer can minimize. It connects the task goal to training by assigning higher scores to worse predictions.',
    intuition: 'Loss is the model receiving a bill for being wrong. Training tries to make the bill smaller.',
    example: 'Cross-entropy loss penalizes a classifier when it assigns low probability to the true class.',
    pitfall: 'Lower training loss is not always better in the real world. It can fall while validation performance gets worse through overfitting.',
  },
  {
    term: 'regularization',
    category: 'Optimization',
    visual: 'loss',
    definition: 'A penalty or constraint that discourages overly complex models.',
    explanation: 'Regularization adds pressure toward simpler, smoother, smaller, or more robust solutions. It is used to improve generalization rather than merely fit the training set.',
    intuition: 'Regularization is a preference for answers that explain the data without memorizing every wrinkle.',
    example: 'L2 regularization discourages very large weights by adding a weight-size penalty to the objective.',
    pitfall: 'Too much regularization can underfit: the model becomes too constrained to capture real structure.',
  },
  {
    term: 'vector',
    category: 'Linear Algebra',
    visual: 'vector',
    definition: 'An ordered list of numbers that represents magnitude, direction, or features.',
    explanation: 'Vectors are the basic containers for model-friendly information. They can represent points, directions, embeddings, hidden states, gradients, or probabilities.',
    intuition: 'A vector is a coordinate address. Similar addresses can mean similar meanings, positions, or behaviors depending on the space.',
    example: 'A word embedding is a vector whose coordinates encode learned semantic and syntactic patterns.',
    pitfall: 'Individual coordinates rarely have obvious meaning by themselves. The geometry of the whole vector space is usually what matters.',
  },
  {
    term: 'matrix',
    category: 'Linear Algebra',
    visual: 'matrix',
    definition: 'A rectangular table of numbers used to transform vectors.',
    explanation: 'Matrices organize many parameters or values at once. Multiplying by a matrix can rotate, scale, project, mix, or map vectors into a new feature space.',
    intuition: 'A matrix is a machine for moving coordinates from one space into another.',
    example: 'A neural network layer often computes a matrix multiply followed by a bias and activation.',
    pitfall: 'Shape matters. Many bugs come from multiplying matrices whose dimensions do not line up.',
  },
  {
    term: 'probability',
    category: 'Probability',
    visual: 'probability',
    definition: 'A number from 0 to 1 describing how likely an event is.',
    explanation: 'Probability quantifies uncertainty. In machine learning, probabilities often represent model belief, data frequency, random sampling behavior, or normalized scores.',
    intuition: 'Probability is uncertainty put on a ruler from impossible to certain.',
    example: 'A classifier might output 0.82 for spam, meaning it assigns high probability to the spam class.',
    pitfall: 'A probability is not always calibrated. A model can say 0.82 and still be systematically overconfident.',
  },
  {
    term: 'representation',
    category: 'Modeling',
    visual: 'vector',
    definition: 'A useful encoding of raw information for computation.',
    explanation: 'A representation changes the form of information so a model or algorithm can use it effectively. Good representations make important structure easier to detect.',
    intuition: 'Representation is translation: the same thing expressed in a language the model can work with.',
    example: 'Text can be represented as token IDs, embeddings, bag-of-words counts, or contextual hidden states.',
    pitfall: 'A representation always keeps some information and discards or distorts other information.',
  },
  {
    term: 'normalization',
    category: 'Neural Networks',
    visual: 'axis',
    definition: 'A rescaling step that makes values easier to compare or optimize.',
    explanation: 'Normalization adjusts values to a more stable scale or distribution. It can improve numerical stability, training speed, and comparability between features or scores.',
    intuition: 'Normalization puts values onto a common measuring stick.',
    example: 'Softmax normalizes logits into probabilities that sum to one.',
    pitfall: 'The normalization axis matters. Normalizing across the wrong dimension changes the meaning of the result.',
  },
  {
    term: 'logits',
    category: 'Neural Networks',
    visual: 'probability',
    definition: 'Raw unnormalized scores before softmax converts them into probabilities.',
    explanation: 'Logits are model scores that can be any real number. Their relative differences matter because softmax turns larger logits into larger probabilities.',
    intuition: 'Logits are votes before the votes are converted into percentages.',
    example: 'If class logits are [4, 1, -2], softmax will heavily favor the first class.',
    pitfall: 'Do not read logits as probabilities. A logit of 4 is not 400 percent confidence.',
  },
  {
    term: 'temperature',
    category: 'Neural Networks',
    visual: 'probability',
    definition: 'A positive scale value that makes softmax probabilities sharper or more diffuse.',
    explanation: 'Temperature divides logits before softmax. Lower temperature exaggerates score differences; higher temperature flattens them and increases randomness.',
    intuition: 'Temperature controls how decisive the probability distribution feels.',
    example: 'In text generation, low temperature repeats safer high-probability tokens; high temperature explores more unusual tokens.',
    pitfall: 'Temperature does not make a model smarter. It changes sampling behavior from the same underlying scores.',
  },
  {
    term: 'attention',
    category: 'Transformers',
    visual: 'attention',
    definition: 'A weighted lookup that lets one item use information from other items.',
    explanation: 'Attention compares a query with keys, turns those comparisons into weights, and mixes values using those weights. This lets each position gather context from relevant positions.',
    intuition: 'Attention is a soft search operation: ask a question, find matching places, blend their contents.',
    example: 'In a sentence, the token "it" can attend strongly to the earlier noun it refers to.',
    pitfall: 'Attention weights are useful clues but not perfect explanations of model reasoning.',
  },
  {
    term: 'token',
    category: 'NLP',
    visual: 'token',
    definition: 'A text unit such as a word, subword, symbol, or character.',
    explanation: 'Tokens are the discrete pieces a text model reads and writes. Tokenization determines how raw text becomes a sequence of model-readable IDs.',
    intuition: 'Tokens are the model alphabet, though the alphabet may contain chunks like "ing" or "token" rather than full words.',
    example: 'The word "unhappiness" might be split into subword tokens like "un", "happi", and "ness".',
    pitfall: 'Tokens are not always words. Counting words and counting tokens can give very different lengths.',
  },
  {
    term: 'embedding',
    category: 'NLP',
    visual: 'vector',
    definition: 'A vector representation that places similar items near each other.',
    explanation: 'Embeddings convert discrete objects such as words, documents, products, or images into vectors. The vector geometry is learned so useful similarities become measurable.',
    intuition: 'An embedding is a location in meaning-space.',
    example: 'Search systems compare a query embedding with document embeddings to retrieve semantically related text.',
    pitfall: 'Nearness depends on training data and objective. Embeddings can encode biases and miss task-specific distinctions.',
  },
  {
    term: 'vocabulary',
    category: 'NLP',
    visual: 'token',
    definition: 'The set of symbols a text model can process directly.',
    explanation: 'A vocabulary maps each token string to an integer ID. Anything outside the vocabulary must be split, mapped to an unknown token, or otherwise handled.',
    intuition: 'The vocabulary is the model dictionary of allowed text pieces.',
    example: 'A tokenizer vocabulary may include common words, punctuation, digits, and subword fragments.',
    pitfall: 'A larger vocabulary can reduce token count but increases embedding table size and may handle rare text unevenly.',
  },
  {
    term: 'query',
    category: 'Transformers',
    visual: 'attention',
    definition: 'A vector asking what information is needed.',
    explanation: 'In attention, the query comes from the position currently gathering context. It is compared against keys to decide which values should influence the output.',
    intuition: 'A query is the search request inside the attention mechanism.',
    example: 'A decoder token uses its query to look back at earlier tokens that help predict the next token.',
    pitfall: 'Query does not mean a user search query here; it is a learned internal vector.',
  },
  {
    term: 'key',
    category: 'Transformers',
    visual: 'attention',
    definition: 'A vector advertising what information a position contains.',
    explanation: 'Keys are compared with queries to compute attention scores. If a key matches a query well, that position receives more attention weight.',
    intuition: 'A key is a label on a memory slot that says what kind of question it can answer.',
    example: 'In self-attention, every token produces a key describing what it can offer to other tokens.',
    pitfall: 'Keys are not dictionary keys with exact lookup. Matching is continuous and learned.',
  },
  {
    term: 'value',
    category: 'Transformers',
    visual: 'attention',
    definition: 'The content vector mixed after attention weights are computed.',
    explanation: 'Values contain the information that actually flows into the attention output. Queries and keys decide weights; values provide the material being blended.',
    intuition: 'If keys are labels, values are the contents behind the labels.',
    example: 'A token may attend to several previous value vectors and combine them into a context-aware representation.',
    pitfall: 'The highest key match is not necessarily the only value used. Attention usually blends multiple values.',
  },
  {
    term: 'head',
    category: 'Transformers',
    visual: 'attention',
    definition: 'One parallel attention channel with its own learned projections.',
    explanation: 'Multi-head attention runs several attention mechanisms side by side. Each head can learn a different way to compare and mix information.',
    intuition: 'Heads are parallel readers looking for different relationships in the same sequence.',
    example: 'One head might track nearby syntax while another tracks long-range references.',
    pitfall: 'Individual heads are not guaranteed to be clean human-readable specialists.',
  },
  {
    term: 'activation',
    category: 'Neural Networks',
    visual: 'axis',
    definition: 'A nonlinear function that lets a network model curved relationships.',
    explanation: 'Activations transform layer outputs before passing them on. Without nonlinear activations, stacked linear layers collapse into one linear transformation.',
    intuition: 'An activation bends the model so it can fit patterns that straight lines cannot.',
    example: 'ReLU outputs zero for negative inputs and leaves positive inputs unchanged.',
    pitfall: 'The activation changes gradient flow. Some choices can saturate or create dead units.',
  },
  {
    term: 'layer',
    category: 'Neural Networks',
    visual: 'graph',
    definition: 'A stage of learned computation inside a network.',
    explanation: 'A layer applies a specific transformation, often using parameters, to move from one representation to another. Networks are built by composing layers.',
    intuition: 'A layer is one workbench in an assembly line of representations.',
    example: 'A transformer block contains attention, normalization, and feed-forward layers.',
    pitfall: 'More layers do not automatically mean better performance; depth can increase cost and optimization difficulty.',
  },
  {
    term: 'backpropagation',
    category: 'Neural Networks',
    visual: 'graph',
    definition: 'The chain-rule procedure for sending gradients backward.',
    explanation: 'Backpropagation computes how each parameter contributed to the loss by moving derivatives backward through the computation graph. It makes large-scale neural network training practical.',
    intuition: 'Backprop is blame assignment through a chain of computations.',
    example: 'After a wrong prediction, backprop determines how much each weight should change to reduce the loss.',
    pitfall: 'Backprop computes gradients; the optimizer decides how to apply them.',
  },
  {
    term: 'computation graph',
    category: 'Neural Networks',
    visual: 'graph',
    definition: 'A directed graph that records how values are computed from inputs to loss.',
    explanation: 'A computation graph breaks a formula or model pass into nodes and edges. It supports automatic differentiation by storing the dependency structure of operations.',
    intuition: 'It is the receipt for how the answer was made, step by step.',
    example: 'For y = (a * b) + c, the graph records multiply first, then add.',
    pitfall: 'The graph is about dependencies, not necessarily about visual neural network layers.',
  },
  {
    term: 'chain rule',
    category: 'Calculus',
    visual: 'graph',
    definition: 'A derivative rule that multiplies local rates of change along a composed path.',
    explanation: 'The chain rule explains how a change in an early variable affects a later result through intermediate operations. Backpropagation is repeated chain rule on a graph.',
    intuition: 'Small effects pass through each link in the chain and get multiplied along the route.',
    example: 'If loss changes with output and output changes with weight, multiply those two rates to get loss change with weight.',
    pitfall: 'For branching graphs, gradients from multiple paths add together.',
  },
  {
    term: 'local derivative',
    category: 'Calculus',
    visual: 'gradient',
    definition: 'The derivative of one operation with respect to its immediate input.',
    explanation: 'A local derivative describes sensitivity at a single operation. Backprop combines many local derivatives to compute gradients for earlier values.',
    intuition: 'It is one tiny slope before that slope is connected to the rest of the graph.',
    example: 'For f(x) = x squared, the local derivative at x = 3 is 6.',
    pitfall: 'A local derivative alone does not tell the whole training update unless it is combined with upstream gradients.',
  },
  {
    term: 'basis',
    category: 'Linear Algebra',
    visual: 'vector',
    definition: 'A coordinate system for describing vectors.',
    explanation: 'A basis is a set of independent directions that can combine to represent any vector in a space. Changing basis changes coordinates without changing the underlying vector.',
    intuition: 'A basis is the set of measuring directions used to describe location.',
    example: 'The usual 2D basis uses one unit arrow right and one unit arrow up.',
    pitfall: 'Different bases can describe the same vector with different coordinate numbers.',
  },
  {
    term: 'objective',
    category: 'Optimization',
    visual: 'loss',
    definition: 'The quantity being optimized or analyzed.',
    explanation: 'An objective formalizes what the system is trying to improve. It may combine a loss term, regularization, constraints, rewards, or other task-specific scores.',
    intuition: 'The objective is the scoreboard the optimizer cares about.',
    example: 'Training may minimize cross-entropy plus a regularization penalty.',
    pitfall: 'Optimizing the wrong objective can produce behavior that scores well but fails the real goal.',
  },
  {
    term: 'derivative',
    category: 'Calculus',
    visual: 'gradient',
    definition: 'The instantaneous rate of change of a function.',
    explanation: 'A derivative measures how much an output changes for a tiny change in an input. It is the one-dimensional version of the sensitivity idea behind gradients.',
    intuition: 'A derivative is the slope at a point, not just the average rise over a wide interval.',
    example: 'If position is a function of time, velocity is its derivative.',
    pitfall: 'A derivative can be zero at flat spots that are not necessarily global optima.',
  },
  {
    term: 'random variable',
    category: 'Probability',
    visual: 'probability',
    definition: 'A numeric outcome of a random process.',
    explanation: 'A random variable assigns numbers to uncertain outcomes so probability tools can analyze them. It can be discrete, like a die roll, or continuous, like measurement error.',
    intuition: 'It is a value you have not observed yet, described by uncertainty.',
    example: 'The number of heads in ten coin flips is a random variable.',
    pitfall: 'A random variable is not random after observation; the uncertainty is in the pre-observation description.',
  },
  {
    term: 'distribution',
    category: 'Probability',
    visual: 'probability',
    definition: 'A rule assigning probabilities across possible outcomes.',
    explanation: 'A distribution describes the full pattern of uncertainty for a random variable. It tells which values are common, rare, or impossible.',
    intuition: 'A distribution is the shape of uncertainty.',
    example: 'A normal distribution places most probability near its mean and less in the tails.',
    pitfall: 'A few samples can look very different from the true distribution, especially with small data.',
  },
  {
    term: 'expectation',
    category: 'Probability',
    visual: 'probability',
    definition: 'The long-run average value of a random variable.',
    explanation: 'Expectation weighs each possible value by its probability. It is the average you would approach after many repeated draws under the same distribution.',
    intuition: 'Expectation is the balance point of a probability distribution.',
    example: 'The expected value of a fair six-sided die is 3.5, even though 3.5 is not a possible roll.',
    pitfall: 'Expected value does not describe risk by itself; spread and tail outcomes also matter.',
  },
  {
    term: 'state',
    category: 'Reinforcement Learning',
    visual: 'graph',
    definition: 'A snapshot of what the agent knows about the environment.',
    explanation: 'A state summarizes the information used to choose an action. In RL, good state design captures what matters for predicting future rewards.',
    intuition: 'State is the agent answering: where am I, and what matters now?',
    example: 'In a grid world, the state may include the agent position and nearby obstacles.',
    pitfall: 'If the state omits important hidden information, the agent may appear inconsistent or confused.',
  },
  {
    term: 'action',
    category: 'Reinforcement Learning',
    visual: 'graph',
    definition: 'A choice the agent can make.',
    explanation: 'An action changes the environment or the agent position within it. Policies map states to actions, either deterministically or probabilistically.',
    intuition: 'Action is the lever available to the agent at the current moment.',
    example: 'A robot action might be move left, grip, or rotate a joint.',
    pitfall: 'The best action depends on future consequences, not only immediate reward.',
  },
  {
    term: 'reward',
    category: 'Reinforcement Learning',
    visual: 'loss',
    definition: 'Feedback that tells the agent how useful an action was.',
    explanation: 'Reward is the signal an RL agent tries to maximize over time. It can be immediate or delayed, sparse or dense, simple or shaped.',
    intuition: 'Reward is the training signal for behavior, not a complete instruction manual.',
    example: 'A game-playing agent may receive positive reward for scoring and negative reward for losing.',
    pitfall: 'Poor reward design can teach shortcuts that satisfy the signal but violate the intended behavior.',
  },
  {
    term: 'hash',
    category: 'Algorithms',
    visual: 'matrix',
    definition: 'A deterministic mapping from data to a compact code.',
    explanation: 'A hash function turns an input into a fixed-size value. Hashes support lookup tables, deduplication, partitioning, and approximate retrieval techniques.',
    intuition: 'A hash is a repeatable fingerprint for data.',
    example: 'A hash table uses a hash of a key to choose where to store or find a value.',
    pitfall: 'Different inputs can collide. Hashes are compact, so they cannot preserve all original information.',
  },
  {
    term: 'graph',
    category: 'Algorithms',
    visual: 'graph',
    definition: 'A set of nodes connected by edges.',
    explanation: 'Graphs model relationships: dependencies, routes, links, transitions, or computation flow. Many ML structures are easiest to understand as graphs.',
    intuition: 'A graph says what things exist and how they connect.',
    example: 'A computation graph connects operations; a social graph connects people.',
    pitfall: 'Graph direction, edge weight, and cycles change the algorithm you need.',
  },
  {
    term: 'iteration',
    category: 'Algorithms',
    visual: 'axis',
    definition: 'One repeated pass of an update rule.',
    explanation: 'Iteration means doing a step, checking the result, then doing another step. Optimization, simulation, and dynamic programming all rely on repeated updates.',
    intuition: 'Iteration is progress by repeated refinement.',
    example: 'Gradient descent performs one parameter update per iteration or batch step.',
    pitfall: 'More iterations can overfit, waste compute, or diverge if the update rule is unstable.',
  },
  {
    term: 'validation set',
    category: 'Core ML',
    visual: 'axis',
    definition: 'Held-out data used to tune model choices before the final test evaluation.',
    explanation: 'A validation set estimates performance during model development. It helps choose hyperparameters, architectures, thresholds, and early stopping points.',
    intuition: 'Validation is the rehearsal audience: you can learn from it, but it is no longer a final blind exam.',
    example: 'You might choose the regularization strength that gives the best validation accuracy.',
    pitfall: 'Repeatedly tuning to validation performance can overfit the validation set.',
  },
  {
    term: 'test set',
    category: 'Core ML',
    visual: 'axis',
    definition: 'Held-out data used once for an honest estimate of generalization.',
    explanation: 'A test set should represent new data and remain untouched until final evaluation. It estimates how the chosen model performs after development decisions are finished.',
    intuition: 'The test set is the sealed exam, not the practice quiz.',
    example: 'After selecting a model using training and validation data, report final accuracy on the test set.',
    pitfall: 'Looking at the test set repeatedly turns it into another validation set.',
  },
  {
    term: 'overfitting',
    category: 'Core ML',
    visual: 'loss',
    definition: 'A failure mode where a model learns training-specific noise instead of reusable pattern.',
    explanation: 'Overfitting happens when training performance improves but generalization worsens. The model captures quirks of the training sample that do not hold for new data.',
    intuition: 'The model memorizes the worksheet instead of learning the subject.',
    example: 'A very deep decision tree may perfectly classify training rows but fail on new examples.',
    pitfall: 'High model capacity is not always bad; the risk depends on data, regularization, and evaluation.',
  },
  {
    term: 'threshold',
    category: 'Core ML',
    visual: 'probability',
    definition: 'A cutoff that turns probabilities or scores into class decisions.',
    explanation: 'A threshold converts continuous model output into an action such as positive or negative. Moving the threshold changes precision, recall, false positives, and false negatives.',
    intuition: 'Threshold is the line where belief becomes a decision.',
    example: 'Classify as fraud when the fraud score is greater than 0.8.',
    pitfall: 'The default 0.5 threshold is not always appropriate, especially with class imbalance or unequal costs.',
  },
  {
    term: 'confusion matrix',
    category: 'Core ML',
    visual: 'confusion',
    definition: 'A table comparing predicted classes with true classes.',
    explanation: 'A confusion matrix breaks classification outcomes into correct and incorrect cells. It reveals which errors the model makes, not just how many.',
    intuition: 'It is a map of where predictions land relative to truth.',
    example: 'For binary classification, the cells are true positives, false positives, false negatives, and true negatives.',
    pitfall: 'Accuracy can hide bad error patterns that the confusion matrix makes visible.',
  },
  {
    term: 'latent',
    category: 'Diffusion',
    visual: 'diffusion',
    definition: 'A compressed hidden representation used for generation.',
    explanation: 'A latent stores important structure in a smaller or more abstract space. Diffusion models often denoise latents instead of full-resolution pixels for efficiency.',
    intuition: 'A latent is a compact sketch the model can work on before decoding to visible output.',
    example: 'Stable Diffusion operates in an image latent space and decodes the final latent into pixels.',
    pitfall: 'Latents are not directly human-readable images, even when they preserve image information.',
  },
  {
    term: 'noise',
    category: 'Diffusion',
    visual: 'diffusion',
    definition: 'Random variation added or removed during generation.',
    explanation: 'Noise is central to diffusion training and sampling. The model learns to predict or remove noise so a random starting point can be transformed into structured data.',
    intuition: 'Noise is the fog that the denoiser gradually clears.',
    example: 'A diffusion sampler starts from noisy latent values and repeatedly denoises them toward an image.',
    pitfall: 'Noise is not just corruption; it provides the generative starting distribution.',
  },
  {
    term: 'scheduler',
    category: 'Diffusion',
    visual: 'diffusion',
    definition: 'A rule that chooses the time or noise steps of a process.',
    explanation: 'A scheduler controls how sampling moves through noise levels. It affects speed, stability, and the path from random noise to generated output.',
    intuition: 'The scheduler is the timetable for denoising.',
    example: 'A sampler may use fewer larger steps for speed or more smaller steps for detail and stability.',
    pitfall: 'A scheduler is not a separate image model; it is a procedure for using the trained denoiser.',
  },
];

export const glossaryTerms = TERM_DETAILS.map((entry) => {
  const slug = slugify(entry.term);
  const category = entry.category;

  return {
    id: slug,
    slug,
    href: `/glossary/${slug}`,
    symbol: SYMBOLS[slug] || entry.term.slice(0, 1),
    aliases: slug === 'temperature' ? ['tau', 'Ï„', 'τ'] : [],
    ...entry,
    image: makeTermImage(entry.term, category, ACCENTS[category], entry.visual),
  };
});

const glossaryById = new Map(glossaryTerms.map((term) => [term.id, term]));
const glossaryByTerm = new Map(glossaryTerms.map((term) => [term.term.toLowerCase(), term]));

export const GLOSSARY_IDS_BY_CATEGORY = {
  nlp: ['token', 'embedding', 'vocabulary', 'vector', 'representation', 'input', 'output', 'concept'],
  transformers: ['attention', 'query', 'key', 'value', 'head', 'vector', 'matrix', 'probability', 'normalization', 'representation'],
  'neural-networks': ['activation', 'layer', 'backpropagation', 'computation-graph', 'chain-rule', 'local-derivative', 'gradient', 'loss', 'parameter', 'logits', 'temperature', 'input', 'output', 'normalization'],
  'advanced-models': ['embedding', 'latent', 'probability', 'representation', 'parameter', 'loss', 'input', 'output'],
  'math-fundamentals': ['matrix', 'vector', 'basis', 'objective', 'derivative', 'gradient', 'iteration', 'concept', 'parameter'],
  'core-ml': ['loss', 'validation-set', 'test-set', 'overfitting', 'threshold', 'confusion-matrix', 'regularization', 'probability', 'parameter', 'objective', 'input', 'output'],
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
