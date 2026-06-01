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

test('frontier architecture overview assessment has 100 curated questions with a focused lab', () => {
  const { quiz, labs } = getLessonAssessment('frontier-llm-architecture-overview');

  assert.equal(quiz.length, 100);
  assert.equal(labs.length, 1);
  assert.deepEqual(labs.map((lab) => lab.id), ['classify-frontier-paper-signals']);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^flao-\d{3}-[a-z0-9-]+$/);
    assert.equal(item.id.startsWith(`flao-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
  }

  const allPositions = quiz.map((item) => item.answerIndex);
  const globalCounts = [0, 1, 2].map((slot) => allPositions.filter((position) => position === slot).length);
  assert.ok(
    Math.max(...globalCounts) - Math.min(...globalCounts) <= 1,
    `answer positions should be globally balanced, got ${globalCounts.join(', ')}`,
  );
});

test('frontier architecture overview assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('frontier architecture overview assessment progresses through the map order', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
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
    ['overview purpose', ['architecture overview lesson']],
    ['dense baseline', ['dense transformer baseline']],
    ['MoE family', ['sparse MoE family']],
    ['attention memory', ['MHA, MQA, GQA, and MLA']],
    ['long context', ['long-context family']],
    ['SSM family', ['recurrent or SSM hybrid']],
    ['diffusion family', ['diffusion language model']],
    ['omni family', ['omni multimodal family']],
    ['paper signals', ['paper signals']],
    ['dense mechanism', ['dense baseline panel']],
    ['attention mechanism', ['MHA cache layout']],
    ['MoE mechanism', ['MoE panel']],
    ['long-context mechanism', ['long-context panel show']],
    ['SSM mechanism', ['SSM or recurrent panel']],
    ['diffusion mechanism', ['single left-to-right next-token loop']],
    ['omni mechanism', ['streams are shown in the omni panel']],
    ['paper application', ['active parameters are far below total parameters']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['production-ready takeaway']],
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

test('frontier architecture overview assessment marks misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
  const misconceptionTerms = [
    /just a larger dense transformer/i,
    /routing and load balance irrelevant/i,
    /MQA gives every query head/i,
    /only a new name for increasing context length/i,
    /larger context window guarantees/i,
    /full pairwise attention matrix as their only memory idea/i,
    /image pixel noise copied into text generation/i,
    /removes token-budget and alignment concerns/i,
    /Active compute and total parameters always mean the same thing/i,
    /single acronym should be treated as a complete architecture audit/i,
    /Thinking budget is the same thing as MoE routing/i,
    /cannot also use attention-memory compression/i,
    /Dense baseline means the model is small or obsolete/i,
    /universally best architecture/i,
    /without creating any new checks/i,
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

test('frontier architecture overview assessment stays within overview lesson scope', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
  const unrelatedScopeLeaks = [
    /\bRLHF\b/i,
    /\bDPO\b/i,
    /\bSFT\b/i,
    /\bLoRA\b/i,
    /\bQLoRA\b/i,
    /\bGRPO\b/i,
    /\btokenization\b/i,
    /\bbackprop\b/i,
    /\boptimizer\b/i,
  ];

  for (const [index, item] of quiz.entries()) {
    const visibleText = `${item.prompt} ${item.choices.join(' ')} ${item.explanation}`;
    assert.ok(!unrelatedScopeLeaks.some((pattern) => pattern.test(visibleText)), `question ${index + 1} leaks unrelated training or preprocessing scope`);
  }
});

test('frontier architecture overview assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const answers = page.map((item) => normalize(correctAnswer(item)));

    for (const [offset, item] of page.entries()) {
      const surroundingQuestions = page
        .filter((_, otherOffset) => otherOffset !== offset)
        .map((other) => normalize(`${other.prompt} ${other.choices.join(' ')}`));
      const leaked = surroundingQuestions.some((visibleText) => visibleText.includes(answers[offset]));
      assert.equal(leaked, false, `${item.id} answer appears in another visible question on same page`);
    }
  }
});

test('frontier architecture overview assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('frontier-llm-architecture-overview');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
