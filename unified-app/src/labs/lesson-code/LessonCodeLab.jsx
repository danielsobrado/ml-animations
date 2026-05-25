import React from 'react';
import { Link } from 'react-router-dom';
import CodeFixLab from '../code-fix/CodeFixLab';
import { getLessonCodeLabExercises } from './lessonCodeLabs.js';

export default function LessonCodeLab({ lessonId }) {
  const exercises = getLessonCodeLabExercises(lessonId);

  if (exercises.length === 0) return null;

  return (
    <section className="ua-lesson-code-lab" id="code-lab" aria-label="Lesson code lab">
      <div className="ua-lesson-code-lab-link">
        <Link to="/labs">Open all code labs</Link>
      </div>
      <CodeFixLab key={lessonId} exercises={exercises} />
    </section>
  );
}
