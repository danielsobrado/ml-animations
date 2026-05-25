import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const NEURAL_NETWORK_LESSON_LABS = createCategoryLessonLabs('neural-networks', {
  kind: 'neural-network computation',
  signalName: 'activation or gradient signal',
  stages: ['forward', 'loss', 'update'],
  stageExplanation: 'Neural-network code is built around forward computation, loss measurement, and parameter updates.',
});
