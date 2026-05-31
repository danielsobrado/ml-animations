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

test('bert assessment has 100 production-ready questions', () => {
  const { quiz } = getLessonAssessment('bert');

  assert.equal(quiz.length, 100);
  assert.equal(new Set(quiz.map((item) => item.id)).size, 100);

  for (const [index, item] of quiz.entries()) {
    assert.match(item.id, /^bert-\d{3}-[a-z0-9-]+$/);
    assert.equal(LEVELS.has(item.level), true, `${item.id} has unexpected level ${item.level}`);
    assert.equal(item.choices.length, 3, `${item.id} should have three choices`);
    assert.ok(item.answerIndex >= 0 && item.answerIndex < 3, `${item.id} answerIndex out of range`);
    assert.ok(item.prompt.length > 20, `${item.id} prompt too short`);
    assert.ok(item.explanation.length > 30, `${item.id} explanation too short`);
    assert.equal(new Set(item.choices.map(normalize)).size, 3, `${item.id} has duplicate choices`);
    assert.equal(item.id.startsWith(`bert-${String(index + 1).padStart(3, '0')}`), true, `${item.id} is out of order`);
  }
});

test('bert assessment avoids duplicate prompts and correct answers', () => {
  const { quiz } = getLessonAssessment('bert');
  const prompts = quiz.map((item) => normalize(item.prompt));
  const answers = quiz.map((item) => normalize(correctAnswer(item)));

  assert.equal(new Set(prompts).size, prompts.length);
  assert.equal(new Set(answers).size, answers.length);
});

test('bert assessment progresses through the lesson learning points', () => {
  const { quiz } = getLessonAssessment('bert');
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
    ['architecture basics', ['encoder only']],
    ['tokenization', ['WordPiece', 'subword']],
    ['embedding sum', ['Token, segment, and position embeddings']],
    ['attention mechanism', ['QK T divided by sqrt']],
    ['pretraining objectives', ['MLM cross entropy']],
    ['fine tuning', ['end to end']],
    ['task application', ['sentiment classification']],
    ['misconception traps', ['statement is false']],
    ['interview synthesis', ['production ready takeaway']],
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

test('bert assessment marks unsafe misconceptions as traps after setup', () => {
  const { quiz } = getLessonAssessment('bert');
  const misconceptionTerms = [
    /BERT reads only left-to-right/i,
    /decoder-only model for prefix generation/i,
    /two separate recurrent passes/i,
    /contains the final label before the encoder runs/i,
    /trains on every token equally/i,
    /Every selected MLM token is replaced by \[MASK\]/i,
    /same as next-token prediction/i,
    /same thing as position embeddings/i,
    /always trains only the new task head/i,
    /free-form autoregressive generator/i,
    /process any length/i,
    /must learn the same pattern/i,
    /automatically align to subword pieces/i,
    /always the best deployment choice/i,
    /alone determines every downstream metric/i,
  ];
  const trapPrompt = /false|rejected|misleading|unsafe|wrong|dangerous|corrected|reject|simplistic|overstates|misses|confuses/i;

  for (const [index, item] of quiz.entries()) {
    const text = `${item.prompt} ${item.choices.join(' ')}`;
    const containsMisconception = misconceptionTerms.some((pattern) => pattern.test(text));
    if (!containsMisconception) continue;

    assert.ok(index >= 75, `${item.id} introduces misconception too early`);
    assert.match(item.prompt, trapPrompt, `${item.id} should mark misconception as a trap`);
  }
});

test('bert assessment avoids visible-page answer leakage', () => {
  const { quiz } = getLessonAssessment('bert');
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

test('bert assessment distributes correct answer positions per page', () => {
  const { quiz } = getLessonAssessment('bert');
  const pageSize = 10;

  for (let pageStart = 0; pageStart < quiz.length; pageStart += pageSize) {
    const page = quiz.slice(pageStart, pageStart + pageSize);
    const counts = [0, 0, 0];
    for (const item of page) counts[item.answerIndex] += 1;
    assert.ok(Math.max(...counts) - Math.min(...counts) <= 1, `imbalanced page at ${pageStart + 1}: ${counts.join(',')}`);
  }
});
