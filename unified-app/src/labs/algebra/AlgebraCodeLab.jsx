import React, { useEffect, useMemo, useState } from 'react';
import CodeFixLab from '../code-fix/CodeFixLab';
import { getAlgebraCodeLabsForLesson } from './algebraLessonCodeLabs';

export default function AlgebraCodeLab({ exercises, lessonId }) {
  const [catalogueExercises, setCatalogueExercises] = useState(null);

  const lessonExercises = useMemo(
    () => (lessonId ? getAlgebraCodeLabsForLesson(lessonId) : null),
    [lessonId],
  );

  useEffect(() => {
    if (exercises || lessonExercises) return undefined;

    let active = true;
    import('./algebraCodeLabs').then(({ ALGEBRA_CODE_LABS }) => {
      if (active) setCatalogueExercises(ALGEBRA_CODE_LABS);
    });

    return () => {
      active = false;
    };
  }, [exercises, lessonExercises]);

  const selectedExercises = exercises
    || lessonExercises
    || catalogueExercises;

  if (!selectedExercises) {
    return (
      <div className="flex items-center justify-center p-12 text-sm text-slate-500">
        Loading code lab...
      </div>
    );
  }

  return <CodeFixLab exercises={selectedExercises} />;
}
