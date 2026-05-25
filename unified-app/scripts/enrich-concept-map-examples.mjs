/**
 * Enriches concept-map leaves missing `example` and removes the REMAINING_MAPS placeholder.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const conceptMapsPath = join(__dirname, '../src/data/conceptMaps.js');

const CURATED_BY_ID = {
  'bias-variance-prereq': 'Ridge with tiny λ overfits; with huge λ validation error rises from underfitting.',
  'penalty-term': 'Logistic loss + λ||w||² updated each SGD step.',
  'train-val-gap': 'Train acc 99%, val acc 71% on the same distribution — large gap.',
  'roc-curve': 'Plot FPR vs TPR as threshold sweeps from 1 to 0.',
  'pr-curve': 'Plot recall vs precision across thresholds.',
  'cross-entropy-loss': 'One-hot y=[0,1,0] with logits z=[1,3,0.5] → CE ≈ 0.41 nats on class 1.',
  'gradient-step': 'θ=2, loss=(θ−3)² → gradient=−2 → θ_next=2.4 with lr=0.2.',
  'train-val-test-honesty': 'Tune on train, pick model on val, report once on test.',
  'reliability-diagram': 'Bin predictions near 0.7; observed positive rate 0.52 → miscalibrated high.',
  'chunk-size-tradeoff': '512-token chunks vs 128-token chunks on the same policy PDF.',
  'vector-index-recall': 'Top-5 retrieval misses the gold paragraph when chunks split mid-sentence.',
  'reranker-cross-encoder': 'Bi-encoder retrieves 50; cross-encoder reranks to top 5 for the prompt.',
  'hallucination-without-context': 'Model cites a statute number absent from retrieved passages.',
  'recall-at-k': 'Gold doc at rank 8 with k=5 → Recall@5 = 0 for that query.',
};

function escapeExample(text) {
  return text.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

function makeExample(body, leafId) {
  if (CURATED_BY_ID[leafId]) return CURATED_BY_ID[leafId];
  const shortMatch = body.match(/short: '((?:\\'|[^'])*)'/);
  const short = shortMatch?.[1]?.replace(/\\'/g, "'") || 'this idea';
  if (body.includes('formula:') && !body.includes('code:')) {
    return `Plug small numbers into the formula for ${short.charAt(0).toLowerCase()}${short.slice(1).replace(/\.$/, '')}.`;
  }
  if (body.includes('code:')) {
    return 'Run the snippet on a toy batch and inspect one row of outputs.';
  }
  if (body.includes('lessonId:')) {
    return `Follow the linked lesson and map ${short.charAt(0).toLowerCase()}${short.slice(1).replace(/\.$/, '')}.`;
  }
  return `Concrete case: ${short.charAt(0).toLowerCase()}${short.slice(1).replace(/\.$/, '')}.`;
}

let source = readFileSync(conceptMapsPath, 'utf8');
source = source.replace(/\n  'REMAINING_MAPS': \{\},\n/, '\n');

let added = 0;
source = source.replace(
  /id: '([^']+)',[\s\S]*?tooltip: tip\(\{([\s\S]*?)\n(\s+)\}\),/g,
  (block, leafId, body, indent) => {
    if (body.includes('example:')) return block;
    const example = makeExample(body, leafId);
    added += 1;
    const exLine = `${indent}  example: '${escapeExample(example)}',\n`;
    const newBody = body.includes('\n              trap:')
      ? body.replace(/\n(\s+)trap:/, `\n${exLine}$1trap:`)
      : `${body}\n${exLine.trimEnd()}`;
    return block.replace(body, newBody);
  },
);

writeFileSync(conceptMapsPath, source);
console.log(JSON.stringify({ added, removedPlaceholder: true }));
