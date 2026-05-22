import React, { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ChevronLeft, ChevronRight, Circle, Eye, FlaskConical } from 'lucide-react';
import { getLessonAssessment, hasAssessmentContent } from '../../data/lessonAssessments';
import {
  getLessonProgress,
  isAssessmentComplete,
  readAssessmentProgress,
  updateLessonProgress,
} from '../../data/learningProgress';

const QUESTIONS_PER_PAGE = 5;

export default function AssessmentPanel({
  lessonId,
  title = 'Lesson check',
  legacyQuestion,
  legacyAnswer,
  legacyExplanation,
}) {
  const assessment = useMemo(() => getLessonAssessment(lessonId), [lessonId]);
  const quizItems = assessment.quiz || [];
  const [quizPage, setQuizPage] = useState(0);
  const [lessonProgress, setLessonProgress] = useState(() => (
    getLessonProgress(readAssessmentProgress(), lessonId)
  ));

  useEffect(() => {
    setQuizPage(0);
    setLessonProgress(getLessonProgress(readAssessmentProgress(), lessonId));
  }, [lessonId]);

  const hasStructuredAssessment = hasAssessmentContent(assessment);
  const hasLegacyCheck = Boolean(legacyQuestion && legacyAnswer);

  if (!hasStructuredAssessment && !hasLegacyCheck) return null;

  const complete = isAssessmentComplete(assessment, lessonProgress);
  const quizPageCount = Math.max(1, Math.ceil(quizItems.length / QUESTIONS_PER_PAGE));
  const activeQuizPage = Math.min(quizPage, quizPageCount - 1);
  const pageStart = activeQuizPage * QUESTIONS_PER_PAGE;
  const pageQuestions = quizItems.slice(pageStart, pageStart + QUESTIONS_PER_PAGE);
  const answeredCount = quizItems.filter((question) => lessonProgress.quiz?.[question.id]?.correct === true).length;

  const persist = (updater) => {
    const nextProgress = updateLessonProgress(lessonId, assessment, updater);
    setLessonProgress(getLessonProgress(nextProgress, lessonId));
  };

  const handleAnswer = (question, selectedIndex) => {
    persist((current) => ({
      ...current,
      quiz: {
        ...(current.quiz || {}),
        [question.id]: {
          selectedIndex,
          correct: selectedIndex === question.answerIndex,
          revealed: true,
        },
      },
    }));
  };

  const handleReveal = (question) => {
    persist((current) => ({
      ...current,
      quiz: {
        ...(current.quiz || {}),
        [question.id]: {
          ...(current.quiz?.[question.id] || {}),
          correct: current.quiz?.[question.id]?.correct === true,
          revealed: true,
        },
      },
    }));
  };

  const handleLabToggle = (lab) => {
    persist((current) => ({
      ...current,
      labs: {
        ...(current.labs || {}),
        [lab.id]: current.labs?.[lab.id] !== true,
      },
    }));
  };

  const handleLegacyReveal = () => {
    persist((current) => ({
      ...current,
      legacyCheck: { revealed: true },
    }));
  };

  return (
    <section className="ua-assessment-panel" aria-label={title}>
      <div className="ua-assessment-head">
        <span>Assessment</span>
        <h2>{title}</h2>
        <p>
          {complete
            ? 'Completed locally.'
            : `${answeredCount}/${quizItems.length} correct. Work through beginner, intermediate, and advanced checks.`}
        </p>
      </div>

      {quizItems.length > QUESTIONS_PER_PAGE && (
        <nav className="ua-assessment-pager" aria-label="Lesson check pages">
          <button
            type="button"
            className="ua-pager-button"
            onClick={() => setQuizPage((value) => Math.max(0, value - 1))}
            disabled={activeQuizPage === 0}
          >
            <ChevronLeft size={15} />
            Prev
          </button>
          <div className="ua-page-dots">
            {Array.from({ length: quizPageCount }, (_, index) => (
              <button
                type="button"
                key={index}
                className={index === activeQuizPage ? 'active' : ''}
                onClick={() => setQuizPage(index)}
                aria-current={index === activeQuizPage ? 'page' : undefined}
              >
                {index + 1}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="ua-pager-button"
            onClick={() => setQuizPage((value) => Math.min(quizPageCount - 1, value + 1))}
            disabled={activeQuizPage === quizPageCount - 1}
          >
            Next
            <ChevronRight size={15} />
          </button>
        </nav>
      )}

      {pageQuestions.map((question, questionIndex) => {
        const state = lessonProgress.quiz?.[question.id] || {};
        const answered = Number.isInteger(state.selectedIndex);
        const revealed = answered || state.revealed;
        const absoluteQuestionIndex = pageStart + questionIndex;

        return (
          <article className="ua-quiz-card" key={question.id}>
            <div className="ua-quiz-meta">
              <span className="ua-quiz-kicker">Question {absoluteQuestionIndex + 1} of {quizItems.length}</span>
              {question.level && <span className="ua-question-level">{question.level}</span>}
            </div>
            <h3>{question.prompt}</h3>
            <div className="ua-choice-list">
              {question.choices.map((choice, choiceIndex) => {
                const selected = state.selectedIndex === choiceIndex;
                const correct = question.answerIndex === choiceIndex;
                const tone = revealed && selected
                  ? selected && correct ? 'correct' : 'incorrect'
                  : revealed && correct ? 'answer' : '';

                return (
                  <button
                    type="button"
                    key={choice}
                    className={`ua-choice-button ${selected ? 'selected' : ''} ${tone}`}
                    onClick={() => handleAnswer(question, choiceIndex)}
                    aria-pressed={selected}
                  >
                    <span>{String.fromCharCode(65 + choiceIndex)}</span>
                    {choice}
                  </button>
                );
              })}
            </div>
            {!revealed && (
              <button type="button" className="ua-reveal-button" onClick={() => handleReveal(question)}>
                <Eye size={15} />
                Reveal answer
              </button>
            )}
            {revealed && (
              <div className={`ua-answer-panel ${state.correct ? 'correct' : ''}`}>
                <strong>{state.correct ? 'Correct.' : `Answer: ${question.choices[question.answerIndex]}`}</strong>
                <p>{question.explanation}</p>
              </div>
            )}
          </article>
        );
      })}

      {quizItems.length > QUESTIONS_PER_PAGE && (
        <nav className="ua-assessment-pager bottom" aria-label="Lesson check pages">
          <button
            type="button"
            className="ua-pager-button"
            onClick={() => setQuizPage((value) => Math.max(0, value - 1))}
            disabled={activeQuizPage === 0}
          >
            <ChevronLeft size={15} />
            Prev
          </button>
          <span>Page {activeQuizPage + 1} of {quizPageCount}</span>
          <button
            type="button"
            className="ua-pager-button"
            onClick={() => setQuizPage((value) => Math.min(quizPageCount - 1, value + 1))}
            disabled={activeQuizPage === quizPageCount - 1}
          >
            Next
            <ChevronRight size={15} />
          </button>
        </nav>
      )}

      {assessment.labs?.length > 0 && (
        <div className="ua-lab-list">
          <div className="ua-quiz-kicker">
            <FlaskConical size={15} />
            Practice labs
          </div>
          {assessment.labs.map((lab) => {
            const done = lessonProgress.labs?.[lab.id] === true;

            return (
              <button
                type="button"
                key={lab.id}
                className={`ua-lab-card ${done ? 'done' : ''}`}
                onClick={() => handleLabToggle(lab)}
                role="checkbox"
                aria-checked={done}
              >
                {done ? <CheckCircle2 size={18} /> : <Circle size={18} />}
                <span>
                  <strong>{lab.title}</strong>
                  <em>{lab.prompt}</em>
                  <small>{lab.successCriteria}</small>
                </span>
              </button>
            );
          })}
        </div>
      )}

      {!hasStructuredAssessment && hasLegacyCheck && (
        <article className="ua-quiz-card">
          <div className="ua-quiz-kicker">Check yourself</div>
          <h3>{legacyQuestion}</h3>
          {!lessonProgress.legacyCheck?.revealed && (
            <button type="button" className="ua-reveal-button" onClick={handleLegacyReveal}>
              <Eye size={15} />
              Reveal answer
            </button>
          )}
          {lessonProgress.legacyCheck?.revealed && (
            <div className="ua-answer-panel correct">
              <strong>{legacyAnswer}</strong>
              {legacyExplanation && <p>{legacyExplanation}</p>}
            </div>
          )}
        </article>
      )}
    </section>
  );
}
