import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const REINFORCEMENT_LEARNING_LESSON_LABS = createCategoryLessonLabs('reinforcement-learning', {
  kind: 'reinforcement-learning loop',
  signalName: 'return or action-value score',
  stages: ['observe', 'act', 'learn'],
  stageExplanation: 'RL loops observe state, choose actions, and update behavior from feedback.',
});
