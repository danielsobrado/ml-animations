import { createCategoryLessonLabs } from '../lessonLabFactory.js';

export const DIFFUSION_LESSON_LABS = createCategoryLessonLabs('diffusion-models', {
  kind: 'diffusion-model pipeline',
  signalName: 'noise or denoising score',
  stages: ['noise', 'condition', 'denoise'],
  stageExplanation: 'Diffusion systems manage noise, conditioning, and denoising as separate implementation stages.',
});
