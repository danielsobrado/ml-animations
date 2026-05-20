import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, ChevronRight, CircleDot, FlaskConical } from 'lucide-react';
import Eq from '../../_design-system/Eq';
import { allAnimations } from '../../data/animations';
import { createLearningModel } from '../../data/animationLearning';

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
      {terms.map((term) => (
        <GlossaryTerm key={term.id} entry={term} />
      ))}
    </span>
  );
}

function MindmapNode({ node, active = false }) {
  if (active) {
    return (
      <div className="ua-map-node active">
        <CircleDot size={14} />
        <span>{node.label}</span>
      </div>
    );
  }

  return (
    <Link to={`/animation/${node.id}`} className="ua-map-node">
      <ChevronRight size={14} />
      <span>{node.label}</span>
    </Link>
  );
}

function MindmapRail({ mindmap }) {
  return (
    <aside className="ua-learning-rail" aria-label="Concept mindmap">
      <div className="ua-learning-rail-head">
        <FlaskConical size={15} />
        <span>Mindmap</span>
      </div>

      <div className="ua-map-group">
        <p>Prereqs</p>
        {mindmap.prereqs.map((node) => (
          <MindmapNode key={node.id} node={node} />
        ))}
      </div>

      <div className="ua-map-connector" />

      <div className="ua-map-group">
        <p>Current</p>
        <MindmapNode node={mindmap.current} active />
      </div>

      <div className="ua-map-connector" />

      <div className="ua-map-group">
        <p>Next</p>
        {mindmap.next.map((node) => (
          <MindmapNode key={node.id} node={node} />
        ))}
      </div>
    </aside>
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
        <section key={card.type} className="ua-learning-card">
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
        {terms.map((term) => (
          <article key={term.id} id={`glossary-${term.slug}`}>
            <img src={term.image.src} alt={term.image.alt} />
            <span>{term.category}</span>
            <h3>
              <Link to={term.href}>{term.term}</Link>
            </h3>
            <p>{term.definition}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function AnimationShell({ animation, children }) {
  const [resetNonce, setResetNonce] = useState(0);
  const model = useMemo(() => createLearningModel(animation, allAnimations), [animation]);

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

      <div className="ua-learning-grid">
        <MindmapRail mindmap={model.mindmap} />

        <main id="math-main-stage" className="ua-main-stage" aria-label={`${animation.name} animation stage`}>
          <div key={resetNonce} className="ua-stage-wrap">
            {children}
          </div>
        </main>

        <LearningCards cards={model.learningCards} />
      </div>

      <Glossary terms={model.glossary} />
    </div>
  );
}
