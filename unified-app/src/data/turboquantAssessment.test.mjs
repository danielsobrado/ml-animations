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
  'mini-turboquant-cache-size',
  'mini-turboquant-dot-products',
  'mini-turboquant-tradeoff',
];

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('turboquant assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('turboquant');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.deepEqual(labs.map((lab) => lab.id), EXPECTED_LAB_IDS);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  const levelCounts = Object.fromEntries([...LEVELS].map((level) => [level, 0]));
  const answerPositionCounts = [0, 0, 0];

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^tq-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`tq-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
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

test('turboquant assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('turboquant');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('turboquant assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('turboquant');
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
    ['KV cache', ['What is stored in the KV cache']],
    ['dot product', ['attention score between a query and a key']],
    ['rotation', ['rotate vectors before quantizing']],
    ['MSE warning', ['minimizing MSE not always enough']],
    ['source scope', ['paper-trail section']],
    ['QJL', ['What does QJL try to correct']],
    ['outliers', ['How does the outlier slider']],
    ['application diagnostics', ['attention top-1 agreement drops']],
    ['online', ['online quantization matter during serving']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['strongest TurboQuant takeaway']],
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

test('turboquant assessment keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('turboquant');
  const misconceptionTerms = [
    /just naive rounding/i,
    /Low MSE guarantees attention rankings are preserved/i,
    /Cached keys do not affect/i,
    /every model and workload can use three bits safely/i,
    /full-precision residual copy/i,
    /removes all quantization error/i,
    /identical failure modes/i,
    /automatically fixes growing per-request KV memory/i,
    /always runs faster/i,
    /same KV-cache geometry/i,
    /after the response is finished/i,
    /effective bits never change/i,
    /toy numbers are universal benchmark claims/i,
    /skip output-quality tests/i,
    /exactly matches FP16 attention/i,
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

test('turboquant assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('turboquant');
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

test('turboquant assessment stays within visible lesson scope', () => {
  const { quiz } = getLessonAssessment('turboquant');
  const outOfScopePatterns = [
    /tokenizer compression/i,
    /shorter prompts/i,
    /training datasets/i,
    /vocabulary size/i,
    /tokenizer merges/i,
    /output length/i,
    /tokenizer-only/i,
    /target-model verification/i,
    /speculative decoding/i,
    /tokenizer vocabulary/i,
    /padding tokens/i,
    /web service/i,
    /only tokenizer/i,
    /answer choices/i,
    /merge order/i,
    /transformer layers/i,
    /accepted speculative tokens/i,
    /prompt template/i,
    /token budget/i,
    /training labels/i,
    /validation labels/i,
    /css bundles/i,
    /source links/i,
    /token sampler/i,
    /prompt font/i,
    /browser layout/i,
    /source link/i,
    /counts source links/i,
    /internet api/i,
    /tokenizer pruning/i,
    /fewer layers/i,
  ];

  for (const item of quiz) {
    const visibleText = `${item.prompt} ${item.choices.join(' ')}`;
    for (const pattern of outOfScopePatterns) {
      assert.equal(pattern.test(visibleText), false, `${item.id} contains out-of-scope clue ${pattern}`);
    }
  }
});

test('turboquant assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('turboquant');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
