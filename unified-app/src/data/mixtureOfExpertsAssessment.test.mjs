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

test('mixture of experts assessment has 100 production-ready questions', () => {
  const { quiz } = getLessonAssessment('moe');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^moe-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`moe-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }
});

test('mixture of experts assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('moe');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('mixture of experts assessment follows the lesson learning order', () => {
  const { quiz } = getLessonAssessment('moe');
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
    ['conditional computation', ['conditional computation']],
    ['router role', ['learned router']],
    ['top-k selection', ['top-k routing']],
    ['load bars', ['Expert Load panel count']],
    ['gating probabilities', ['gating probabilities represent']],
    ['dispatch', ['dispatch mean']],
    ['combine selected outputs', ['selected outputs are combined']],
    ['load imbalance', ['load imbalance harmful']],
    ['active versus total', ['active parameters']],
    ['application diagnosis', ['per-expert load distribution']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['production-ready MoE takeaway']],
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

test('mixture of experts assessment marks misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('moe');
  const misconceptionTerms = [
    /runs every expert for every token/i,
    /pure random selection/i,
    /always means exactly one expert/i,
    /forces every token to use all added experts/i,
    /Load balance no longer matters/i,
    /Inactive experts require no storage/i,
    /All tokens in a batch must share/i,
    /guaranteed just by naming boxes experts/i,
    /directly emits the final generated text/i,
    /Active parameters always equal total parameters/i,
    /Top-2 halves expert computation/i,
    /every expert to learn identical weights/i,
    /same routing behavior/i,
    /trained routers ignore token content/i,
    /removes routing risks/i,
  ];
  const trapPrompt = /false|rejected|unsafe|wrong|dangerous|misleading|contradicts|overstates/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('mixture of experts assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('moe');
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

test('mixture of experts assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('moe');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
