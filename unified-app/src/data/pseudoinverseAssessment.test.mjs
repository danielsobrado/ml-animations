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

test('pseudoinverse assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('pseudoinverse');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^pinv-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`pinv-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('pseudoinverse assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('pseudoinverse');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('pseudoinverse assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('pseudoinverse');
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
    ['purpose', ['main purpose of the pseudoinverse']],
    ['full column', ['A+ = (A^T A)^-1 A^T']],
    ['full row', ['A+ = A^T (A A^T)^-1']],
    ['rank deficient', ['A+ = V Sigma+ U^T']],
    ['SVD path', ['displayed SVD-style pathway']],
    ['zero practice', ['zero singular value']],
    ['mechanism', ['Why does the SVD rule handle rank deficiency']],
    ['application', ['A tall matrix has independent columns']],
    ['misconception traps', ['Which statement is false about the pseudoinverse']],
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

test('pseudoinverse keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('pseudoinverse');
  const trapIds = [
    'pinv-076-false-ordinary-inverse',
    'pinv-077-wrong-zero-reciprocal',
    'pinv-078-dangerous-infinity',
    'pinv-079-false-column-outcome',
    'pinv-080-false-row-outcome',
    'pinv-081-wrong-rank-deficient',
    'pinv-082-false-formula-choice',
    'pinv-083-dangerous-fit',
    'pinv-084-wrong-svd-order',
    'pinv-085-false-small-zero',
    'pinv-086-wrong-lost-info',
    'pinv-087-false-transpose',
    'pinv-088-dangerous-one-rule',
    'pinv-089-false-svd-square',
    'pinv-090-wrong-title-only',
  ];
  const misconceptionTerms = [
    /only works for square full-rank matrices/i,
    /replaced by 1\/0/i,
    /Use infinity/i,
    /many exact solutions by minimum norm/i,
    /unique least-squares solution because columns/i,
    /all singular directions can be inverted safely/i,
    /without checking rank or shape/i,
    /always means the original equation was exactly solved/i,
    /skips U\^T b/i,
    /small positive singular value is exactly the same as zero/i,
    /recovers information from directions A collapses/i,
    /decorative and meaningless/i,
    /full-column formula for every matrix/i,
    /only works when A is square and invertible/i,
    /title alone tells you/i,
  ];
  const trapPrompt = /false|misleading|wrong|dangerous|rejected|unsafe/i;

  assert.deepEqual(quiz.slice(75, 90).map((question) => question.id), trapIds);

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('pseudoinverse assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('pseudoinverse');
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

test('pseudoinverse assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('pseudoinverse');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
