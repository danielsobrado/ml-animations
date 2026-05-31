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

test('least squares projection assessment has 100 curated questions', () => {
  const { quiz, labs } = getLessonAssessment('least-squares-projection');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 3);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^lsp-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`lsp-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('least squares projection assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('least-squares-projection');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('least squares projection assessment progresses through lesson learning points', () => {
  const { quiz } = getLessonAssessment('least-squares-projection');
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
    ['purpose', ['least-squares projection solve']],
    ['column space', ['What does Col(A) represent']],
    ['projection', ['What does b_hat represent']],
    ['residual', ['What is the residual e']],
    ['normal equation', ['A^T A x_hat = A^T b']],
    ['objective', ['What objective is least squares minimizing']],
    ['derive', ['derivation path']],
    ['cases', ['near, far, and exactly consistent']],
    ['application', ['given x_hat']],
    ['misconception traps', ['Which statement is false about the least-squares residual']],
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

test('least squares projection keeps misconception traps after setup', () => {
  const { quiz } = getLessonAssessment('least-squares-projection');
  const trapIds = [
    'lsp-076-false-residual-inside',
    'lsp-077-wrong-maximize',
    'lsp-078-false-normal-ax0',
    'lsp-079-wrong-atyb',
    'lsp-080-false-zero-always',
    'lsp-081-wrong-small-exact',
    'lsp-082-false-large-no-projection',
    'lsp-083-dangerous-normal-exact',
    'lsp-084-wrong-bhat-residual',
    'lsp-085-false-e-parallel',
    'lsp-086-wrong-col-all-space',
    'lsp-087-false-transpose-purpose',
    'lsp-088-dangerous-ignore-residual',
    'lsp-089-false-farthest',
    'lsp-090-wrong-title-only',
  ];
  const misconceptionTerms = [
    /residual is always inside Col\(A\)/i,
    /maximizes \|\|Ax - b\|\|/i,
    /A x = 0/i,
    /A\^T y = b/i,
    /zero for every least-squares problem/i,
    /small residual proves exact consistency/i,
    /large residual means no projection exists/i,
    /always means Ax_hat equals b exactly/i,
    /b_hat is the residual vector/i,
    /e is parallel to Col\(A\)/i,
    /always contains every possible b/i,
    /only to make the equation look symmetric/i,
    /without checking residual size/i,
    /farthest point in Col\(A\)/i,
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

test('least squares projection assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('least-squares-projection');
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

test('least squares projection assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('least-squares-projection');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
