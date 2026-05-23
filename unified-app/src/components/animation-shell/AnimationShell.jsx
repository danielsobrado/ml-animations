import React, { Suspense, lazy, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen } from 'lucide-react';
import Eq from '../../_design-system/Eq';
import { allAnimations } from '../../data/animations';
import { createLearningModel } from '../../data/animationLearning';
import { getLessonAssessment, hasAssessmentContent } from '../../data/lessonAssessments';
import AssessmentPanel from './AssessmentPanel';

const ConceptMindmap = lazy(() => import('./ConceptMindmap'));

function GlossaryTerm({ entry }) {
  const [open, setOpen] = useState(false);

  if (!entry) return null;

  return (
    <span className="ua-term-wrap">
      <button
        type="button"
        className="ua-term"
        data-term-id={entry.id}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        {entry.term}
      </button>
      {open && (
        <span className="ua-term-popover" role="dialog">
          <img src={entry.image.src} alt={entry.image.alt} />
          <strong>{entry.term}</strong>
          <span>{entry.definition}</span>
          <Link to={entry.href}>Open glossary note</Link>
        </span>
      )}
    </span>
  );
}

function GlossaryTermList({ terms }) {
  if (!terms?.length) return null;

  return (
    <span className="ua-term-list">
      {terms.map((term, index) => (
        <GlossaryTerm key={`${term.id}-${index}`} entry={term} />
      ))}
    </span>
  );
}

function MindmapFallback() {
  return (
    <section className="ua-concept-map" aria-label="Concept mindmap">
      <div className="ua-learning-rail-head">
        <span>Mindmap</span>
      </div>
      <div className="ua-map-canvas ua-map-loading">Loading mindmap</div>
    </section>
  );
}

function LearningCards({ cards }) {
  return (
    <aside className="ua-card-stack" aria-label="Learning cards">
      <div className="ua-learning-rail-head">
        <BookOpen size={15} />
        <span>Learning Cards</span>
      </div>
      {cards.map((card) => (
        <section key={card.type} className={`ua-learning-card ${card.type}`}>
          <div className="ua-learning-card-head">
            <span>{card.label}</span>
            <h3>{card.title}</h3>
          </div>
          <p>{card.body}</p>
          {card.equation && (
            <div className="ua-card-equation">
              <Eq tex={card.equation} />
            </div>
          )}
          <GlossaryTermList terms={card.terms} />
        </section>
      ))}
    </aside>
  );
}

function MathControls({ model, onReset, onFocusStage }) {
  const [prereq] = model.mindmap.prereqs;
  const [next] = model.mindmap.next;

  const actions = {
    prereq: prereq ? { as: Link, to: `/animation/${prereq.id}` } : { as: 'button', onClick: onFocusStage },
    reset: { as: 'button', onClick: onReset },
    play: { as: 'button', onClick: onFocusStage },
    sum: { as: 'a', href: '#math-glossary' },
    next: next ? { as: Link, to: `/animation/${next.id}` } : { as: 'button', onClick: onFocusStage },
  };

  return (
    <nav className="ua-math-controls" aria-label="Math animation controls">
      {model.controls.map((control) => {
        const action = actions[control.id];
        const Component = action.as;
        const props = { ...action };
        delete props.as;

        return (
          <Component key={control.id} {...props} className="ua-sigil-button" data-math-control="true">
            <span>{control.sigil}</span>
            {control.label}
          </Component>
        );
      })}
    </nav>
  );
}

function Glossary({ terms }) {
  return (
    <section id="math-glossary" className="ua-glossary-panel">
      <div className="ua-glossary-head">
        <span>Glossary</span>
        <h2>Terms in this animation</h2>
      </div>
      <div className="ua-glossary-grid">
        {terms.map((term, index) => (
          <article key={`${term.id}-${index}`} id={`glossary-${term.slug}`}>
            <img src={term.image.src} alt={term.image.alt} />
            <span>{term.category}</span>
            <h3>
              <Link to={term.href}>{term.term}</Link>
            </h3>
            {(term.symbol || term.aliases?.length > 0) && (
              <div className="ua-glossary-card-meta" aria-label={`${term.term} metadata`}>
                {term.symbol && term.symbol !== term.term.slice(0, 1) && <span>{term.symbol}</span>}
                {term.aliases?.slice(0, 3).map((alias) => (
                  <span key={alias}>{alias}</span>
                ))}
                {term.aliases?.length > 3 && (
                  <span>+{term.aliases.length - 3} more</span>
                )}
              </div>
            )}
            <p>{term.definition}</p>
            {term.intuition && <p className="ua-glossary-intuition">{term.intuition}</p>}
          </article>
        ))}
      </div>
    </section>
  );
}

export default function AnimationShell({ animation, children }) {
  const [resetNonce, setResetNonce] = useState(0);
  const model = useMemo(() => createLearningModel(animation, allAnimations), [animation]);
  const assessment = useMemo(() => getLessonAssessment(animation.id), [animation.id]);
  const showShellAssessment = hasAssessmentContent(assessment) && animation.categoryId !== 'core-ml';

  const resetStage = () => {
    setResetNonce((value) => value + 1);
    document.getElementById('math-main-stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const focusStage = () => {
    document.getElementById('math-main-stage')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  return (
    <div className="ua-learning-shell">
      <header className="ua-learning-strip">
        <div>
          <span>{model.chips.category}</span>
          <h2>{model.conceptName}</h2>
        </div>
        <div className="ua-headline-eq">
          <Eq tex={model.headlineEquation.latex} />
        </div>
        <div className="ua-chip-row">
          <span>{model.chips.difficulty}</span>
          <span>{model.chips.minutes}</span>
          <span>{model.chips.prereq}</span>
        </div>
      </header>

      <MathControls model={model} onReset={resetStage} onFocusStage={focusStage} />

      <Suspense fallback={<MindmapFallback />}>
        <ConceptMindmap mindmap={model.mindmap} />
      </Suspense>

      <div className="ua-learning-grid">
        <main id="math-main-stage" className="ua-main-stage" aria-label={`${animation.name} animation stage`}>
          <div key={resetNonce} className="ua-stage-wrap">
            {children}
          </div>
        </main>

        <LearningCards cards={model.learningCards} />
      </div>

      {showShellAssessment && (
        <AssessmentPanel lessonId={animation.id} title={`${animation.name} check`} />
      )}

      <Glossary terms={model.glossary} />
    </div>
  );
}
