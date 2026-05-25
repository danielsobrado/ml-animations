import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const EXPERIMENTATION_LESSON_LABS = createCategoryLessonLabs('experimentation-causal-ml', {
  kind: 'experiment analysis',
  signalName: 'effect or balance score',
  stages: ['assign', 'measure', 'compare'],
  stageExplanation: 'Experiment code must separate assignment, measurement, and comparison to support causal claims.',
});
