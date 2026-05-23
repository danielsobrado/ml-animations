import React from 'react';

const STAT_TONES = {
  slate: 'ua-lesson-stat-neutral',
  cyan: 'ua-lesson-stat-cool',
  emerald: 'ua-lesson-stat-good',
  amber: 'ua-lesson-stat-warn',
  rose: 'ua-lesson-stat-risk',
};

export function LessonStage({ children, className = '' }) {
  return <div className={`ua-lesson-stage ${className}`.trim()}>{children}</div>;
}

export function LessonPanel({ children, className = '', compact = false }) {
  return (
    <section className={`ua-lesson-panel${compact ? ' ua-lesson-panel-compact' : ''} ${className}`.trim()}>
      {children}
    </section>
  );
}

export function LessonKicker({ icon: Icon, children }) {
  return (
    <div className="ua-lesson-kicker">
      {Icon ? <Icon size={16} /> : null}
      {children}
    </div>
  );
}

export function LessonStat({ label, value, detail, tone = 'slate' }) {
  return (
    <div className={`ua-lesson-stat ${STAT_TONES[tone] || STAT_TONES.slate}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function LessonResetButton({ onClick, children = 'Reset' }) {
  return (
    <button type="button" onClick={onClick} className="ua-lesson-reset">
      {children}
    </button>
  );
}

export function LessonCallout({ tone = 'neutral', children }) {
  const toneClass = {
    neutral: 'ua-lesson-callout-neutral',
    good: 'ua-lesson-callout-good',
    warn: 'ua-lesson-callout-warn',
    risk: 'ua-lesson-callout-risk',
  }[tone] || 'ua-lesson-callout-neutral';

  return <div className={`ua-lesson-callout ${toneClass}`}>{children}</div>;
}

export function LessonEquation({ children }) {
  return <div className="ua-lesson-equation">{children}</div>;
}
