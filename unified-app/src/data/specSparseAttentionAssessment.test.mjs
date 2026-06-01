import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);
const EXPECTED_LEVEL_COUNTS = {
  Foundation: 20,
  Mechanism: 30,
  Application: 25,
  Tricky: 15,
  Interview: 10,
};
const EXPECTED_LAB_IDS = [
  'mini-spec-sparse-prefix',
  'mini-spec-sparse-criticality',
  'mini-spec-sparse-schedule',
];

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('spec sparse attention assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('spec-sparse-attention');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), EXPECTED_LAB_IDS);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  const levelCounts = Object.fromEntries([...LEVELS].map((level) => [level, 0]));
  const answerPositionCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^specsa-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`specsa-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    levelCounts[item.level] += 1;
    answerPositionCounts[item.answerIndex] += 1;
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }

  assert.deepEqual(levelCounts, EXPECTED_LEVEL_COUNTS);
  assert.ok(Math.max(...answerPositionCounts) - Math.min(...answerPositionCounts) <= 1, `imbalanced answers: ${answerPositionCounts.join(',')}`);
});

test('spec sparse attention assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('spec sparse attention assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const ranges = [
    ['Foundation', 0, 20],
    ['Mechanism', 20, 50],
    ['Application', 50, 75],
    ['Tricky', 75, 90],
    ['Interview', 90, 100],
  ];

  for (const [level, start, end] of ranges) {
    assert.equal(quiz.slice(start, end).every((item) => item.level === level), true, `${level} range mismatch`);
  }

  const milestones = [
    ['two bottlenecks', ['two bottlenecks combined']],
    ['accepted prefix', ['longest draft-token prefix']],
    ['core mismatch', ['central mismatch']],
    ['SpecAttn', ['self-speculative sparse drafting']],
    ['SpecSA', ['sparse speculative-verification framework']],
    ['exact merged', ['exact merged scheduling']],
    ['shared index', ['approximate shared-index scheduling']],
    ['refresh reuse', ['refresh and reuse layers']],
    ['planner', ['lesson need a planner']],
    ['application audit', ['product team claims a 2x speedup']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['strongest final takeaway']],
  ];

  let previous = -1;
  for (const [name, terms] of milestones) {
    const index = quiz.findIndex((item) => (
      terms.every((term) => normalize(`${item.prompt} ${item.choices.join(' ')} ${item.explanation}`).includes(normalize(term)))
    ));
    assert.notEqual(index, -1, `missing milestone: ${name}`);
    assert.ok(index > previous, `${name} appears out of order`);
    previous = index;
  }
});

test('spec sparse attention assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const misconceptionTerms = [
    /automatically multiply in every workload/i,
    /accepted without full-attention verification/i,
    /always improves end-to-end throughput/i,
    /Use only the last accepted token attention pattern/i,
    /every verifier row logit can be ignored/i,
    /attend to the whole union without masks/i,
    /preserves exact per-query sparse layouts/i,
    /recompute selected indices from scratch every time/i,
    /fixed for all prompts and context regimes/i,
    /always select identical KV blocks/i,
    /automatically optimal for verification/i,
    /no reuse because every query is independent/i,
    /regardless of acceptance or verification cost/i,
    /without measuring quality or acceptance impact/i,
    /prove exact paper benchmark numbers/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('spec sparse attention assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingVisibleText = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(`${other.prompt} ${other.choices.join(' ')}`));
      const leaked = surroundingVisibleText.some((text) => text.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears elsewhere on same page`);
    }
  }
});

test('spec sparse attention assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const outOfScopePatterns = [
    /tokenization rules/i,
    /vocabulary size/i,
    /tokenizer must retrain/i,
    /output embedding/i,
    /static html/i,
    /vocabulary items/i,
    /transformer layers at deployment/i,
    /new vocabulary item/i,
    /fixed vocabulary mask/i,
    /replacement tokenizer/i,
    /supervised fine-tuning objective/i,
    /rl rewards/i,
    /model weights/i,
    /optimizer state/i,
    /english words/i,
    /tokenizer changes/i,
    /final answers the quiz/i,
    /english vocabulary/i,
    /retrained verifier/i,
    /tokenizer-only assistant/i,
    /vocabulary logits/i,
    /vocabulary id/i,
    /class labels/i,
    /layer weight/i,
    /cpu memory/i,
    /tokenizer language/i,
    /answer choices/i,
    /cpu tokenizers/i,
    /printed in the ui/i,
    /rejected answers in the quiz/i,
    /train the base model/i,
    /generated text to disk/i,
    /numeric id only/i,
    /gpu temperature/i,
    /failed unit tests/i,
    /ui buttons/i,
    /tokenizer setup/i,
    /dense matrix multiplication/i,
    /tokenizer optimization/i,
    /second tokenizer/i,
    /natural language vocabulary/i,
    /changes tokenization/i,
    /changes labels/i,
    /screenshot/i,
  ];

  for (const item of quiz) {
    const visibleText = `${item.prompt} ${item.choices.join(' ')}`;
    for (const pattern of outOfScopePatterns) {
      assert.equal(pattern.test(visibleText), false, `${item.id} contains out-of-scope clue ${pattern}`);
    }
  }
});

test('spec sparse attention assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('spec-sparse-attention');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
