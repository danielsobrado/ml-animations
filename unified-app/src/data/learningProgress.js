export const ASSESSMENT_PROGRESS_KEY = 'ml-animations:assessment-progress:v1';
export const COMPLETED_LESSONS_KEY = 'ml-animations:completed-lessons:v1';
export const LEARNING_PROGRESS_EVENT = 'ml-animations:progress-updated';
const EMPTY_ASSESSMENT = Object.freeze({
  quiz: Object.freeze([]),
  labs: Object.freeze([]),
});
export const DEFAULT_COMPLETION_POLICY = Object.freeze({
  quickCheckRequired: 5,
  masteryRequired: 12,
  passThreshold: 0.8,
  labsRequired: 1,
  strategyReviewOptional: true,
});

function getStorage(storage) {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;
  return window.localStorage;
}

function parseJson(value, fallback) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function notifyProgressUpdated() {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new Event(LEARNING_PROGRESS_EVENT));
}

export function readAssessmentProgress(storage) {
  const target = getStorage(storage);
  if (!target) return {};
  return parseJson(target.getItem(ASSESSMENT_PROGRESS_KEY), {});
}

export function writeAssessmentProgress(progress, storage) {
  const target = getStorage(storage);
  if (!target) return;
  target.setItem(ASSESSMENT_PROGRESS_KEY, JSON.stringify(progress));
}

export function readCompletedLessons(storage) {
  const target = getStorage(storage);
  if (!target) return new Set();
  const lessonIds = parseJson(target.getItem(COMPLETED_LESSONS_KEY), []);
  return new Set(Array.isArray(lessonIds) ? lessonIds : []);
}

export function writeCompletedLessons(completedLessons, storage) {
  const target = getStorage(storage);
  if (!target) return;
  target.setItem(COMPLETED_LESSONS_KEY, JSON.stringify([...completedLessons]));
}

export function getLessonProgress(progress, lessonId) {
  return progress[lessonId] || { quiz: {}, labs: {}, legacyCheck: null };
}

export function getCompletionPolicy(assessment = EMPTY_ASSESSMENT) {
  return {
    ...DEFAULT_COMPLETION_POLICY,
    ...(assessment.completionPolicy || {}),
  };
}

export function getCompletionStatus(assessment = EMPTY_ASSESSMENT, lessonProgress = {}) {
  const quizItems = assessment.quiz || [];
  const labItems = assessment.labs || [];
  const policy = getCompletionPolicy(assessment);
  const coreQuestions = quizItems
    .filter((item) => item.countsForCompletion !== false)
    .slice(0, policy.masteryRequired);
  const requiredQuestionCount = Math.min(policy.masteryRequired, coreQuestions.length);
  const requiredLabCount = Math.min(policy.labsRequired, labItems.length);
  const correctCoreCount = coreQuestions.filter((item) => lessonProgress.quiz?.[item.id]?.correct === true).length;
  const completedLabCount = labItems.filter((item) => lessonProgress.labs?.[item.id] === true).length;
  const requiredCorrectCount = Math.ceil(requiredQuestionCount * policy.passThreshold);
  const quizComplete = requiredQuestionCount === 0 || correctCoreCount >= requiredCorrectCount;
  const labsComplete = requiredLabCount === 0 || completedLabCount >= requiredLabCount;

  return {
    policy,
    coreQuestions,
    requiredQuestionCount,
    requiredCorrectCount,
    correctCoreCount,
    requiredLabCount,
    completedLabCount,
    quizComplete,
    labsComplete,
    complete: quizComplete && labsComplete,
  };
}

export function isAssessmentComplete(assessment = EMPTY_ASSESSMENT, lessonProgress = {}) {
  const quizItems = assessment.quiz || [];
  const labItems = assessment.labs || [];
  const hasStructuredItems = quizItems.length > 0 || labItems.length > 0;

  if (!hasStructuredItems) {
    return Boolean(lessonProgress.legacyCheck?.revealed);
  }

  return getCompletionStatus(assessment, lessonProgress).complete;
}

export function reconcileCompletedLesson(lessonId, assessment, progress, storage) {
  const lessonProgress = getLessonProgress(progress, lessonId);
  const completedLessons = readCompletedLessons(storage);

  if (isAssessmentComplete(assessment, lessonProgress)) {
    completedLessons.add(lessonId);
  } else {
    completedLessons.delete(lessonId);
  }

  writeCompletedLessons(completedLessons, storage);
  return completedLessons;
}

export function updateLessonProgress(lessonId, assessment, updater, storage) {
  const progress = readAssessmentProgress(storage);
  const currentLessonProgress = getLessonProgress(progress, lessonId);
  const nextLessonProgress = updater(currentLessonProgress);
  const nextProgress = {
    ...progress,
    [lessonId]: nextLessonProgress,
  };

  writeAssessmentProgress(nextProgress, storage);
  reconcileCompletedLesson(lessonId, assessment, nextProgress, storage);
  notifyProgressUpdated();
  return nextProgress;
}
