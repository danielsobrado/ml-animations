import assert from 'node:assert/strict';
import test from 'node:test';
import { getLessonAssessment } from './lessonAssessments.js';

const LEVELS = new Set(['Foundation', 'Mechanism', 'Application', 'Tricky', 'Interview']);

function normalize(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function correctAnswer(item) {
  return item.choices[item.answerIndex];
}

test('condition number assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('condition-number');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^cond-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`cond-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('condition number assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('condition-number');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('condition number assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('condition-number');
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
    ['purpose', ['main purpose of condition number']],
    ['visual', ['unit circle']],
    ['formula', ['kappa(A) = sigma_max / sigma_min']],
    ['near singular lab', ['sigma_min approaches zero']],
    ['code lab', ['condition-number code lab']],
    ['mechanism', ['ellipse axes']],
    ['application', ['conditionNumber([5, 4, 2])']],
    ['misconception traps', ['Which condition-number claim is false']],
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

test('condition number keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('condition-number');
  const trapIds = [
    'cond-076-false-high-good',
    'cond-077-wrong-ratio',
    'cond-078-false-zero-finite',
    'cond-079-dangerous-det-only',
    'cond-080-false-normal-equations',
    'cond-081-wrong-residual-certainty',
    'cond-082-false-one-vector',
    'cond-083-wrong-scaling',
    'cond-084-false-formula-stability',
    'cond-085-dangerous-norm-mix',
    'cond-086-false-universal-threshold',
    'cond-087-wrong-free-regularization',
    'cond-088-false-near-singular-unsolvable',
    'cond-089-dangerous-rounded-display',
    'cond-090-wrong-largest-only',
  ];
  const misconceptionTerms = [
    /higher kappa always means a better-conditioned/i,
    /sigma_min \/ sigma_max/i,
    /finite safe number/i,
    /Use determinant alone as a complete replacement/i,
    /always improves conditioning/i,
    /tiny residual alone proves/i,
    /one random vector is enough/i,
    /Feature scaling is irrelevant/i,
    /guarantees floating-point stability/i,
    /without checking which norm/i,
    /One universal kappa cutoff/i,
    /no bias or tradeoff/i,
    /always means the system has no possible approximate use/i,
    /rounded preset readout as an exact proof/i,
    /Only sigma_max matters/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|unsafe/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('condition number assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('condition-number');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingPrompts = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(other.prompt));
      const leaked = surroundingPrompts.some((prompt) => prompt.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another prompt on same page`);
    }
  }
});

test('condition number assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('condition-number');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
