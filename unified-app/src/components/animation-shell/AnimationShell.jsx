import React, { Suspense, lazy, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, BookOpen, FileSearch, GitCompare, ShieldCheck } from 'lucide-react';
import Eq from '../../_design-system/Eq';
import { allAnimations } from '../../data/animations';
import { createLearningModel } from '../../data/animationLearning';
import { getLessonAssessment, hasAssessmentContent } from '../../data/lessonAssessments';
import AssessmentPanel from './AssessmentPanel';

const ConceptMindmap = lazy(() => import('./ConceptMindmap'));
const GLOSSARY_PAGE_SIZE = 15;

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

function DepthList({ title, items }) {
  if (!items?.length) return null;

  return (
    <div>
      <h4>{title}</h4>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function ConceptComparisonCards({ comparisons }) {
  if (!comparisons?.length) return null;

  return (
    <section className="ua-depth-panel" aria-label="Concept comparisons">
      <div className="ua-depth-panel-head">
        <GitCompare size={17} />
        <span>Concept comparisons</span>
      </div>
      <div className="ua-depth-grid">
        {comparisons.map((comparison) => (
          <article key={comparison.id} className="ua-depth-card ua-comparison-card">
            <span>{comparison.title}</span>
            <div>
              <strong>{comparison.left}</strong>
              <p>{comparison.leftSummary}</p>
            </div>
            <div>
              <strong>{comparison.right}</strong>
              <p>{comparison.rightSummary}</p>
            </div>
            <p><b>Common mistake:</b> {comparison.commonMistake}</p>
            <p><b>Diagnostic:</b> {comparison.diagnostic}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function FailureGallery({ failures }) {
  if (!failures?.length) return null;

  return (
    <section className="ua-depth-panel" aria-label="Failure gallery">
      <div className="ua-depth-panel-head">
        <AlertTriangle size={17} />
        <span>Failure gallery</span>
      </div>
      <div className="ua-depth-grid">
        {failures.map((failure) => (
          <article key={failure.id} className="ua-depth-card">
            <span>{failure.track}</span>
            <h3>{failure.title}</h3>
            <p><b>Symptom:</b> {failure.symptom}</p>
            <p><b>Why:</b> {failure.whyItHappens}</p>
            <p><b>Detect:</b> {failure.howToDetect}</p>
            <p><b>Fix:</b> {failure.howToFix}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function PaperReadingMode({ signals }) {
  if (!signals?.length) return null;

  return (
    <section className="ua-depth-panel" aria-label="Paper reading mode">
      <div className="ua-depth-panel-head">
        <FileSearch size={17} />
        <span>Paper-reading mode</span>
      </div>
      <div className="ua-depth-grid">
        {signals.map((signal) => (
          <article key={signal.id} className="ua-depth-card">
            <span>When a paper says</span>
            <h3>{signal.phrase}</h3>
            <DepthList title="Ask" items={signal.ask} />
            <p><b>Means:</b> {signal.means}</p>
            <p><b>Does not mean:</b> {signal.doesNotMean}</p>
            <DepthList title="Check" items={signal.check} />
          </article>
        ))}
      </div>
    </section>
  );
}

function CaveatBoxes({ caveats }) {
  if (!caveats?.length) return null;

  return (
    <section className="ua-depth-panel" aria-label="Caveats and boundaries">
      <div className="ua-depth-panel-head">
        <ShieldCheck size={17} />
        <span>What this does not solve</span>
      </div>
      <div className="ua-depth-grid">
        {caveats.map((caveat) => (
          <article key={caveat.id} className="ua-depth-card">
            <DepthList title="Solves" items={caveat.solves} />
            <DepthList title="Does not solve" items={caveat.doesNotSolve} />
            <DepthList title="Can go wrong" items={caveat.canGoWrong} />
            <DepthList title="How to test it" items={caveat.howToTest} />
          </article>
        ))}
      </div>
    </section>
  );
}

function CurriculumDepthPanels({ depth }) {
  const hasDepth = depth && (
    depth.comparisons.length > 0
    || depth.failures.length > 0
    || depth.paperSignals.length > 0
    || depth.caveats.length > 0
  );

  if (!hasDepth) return null;

  return (
    <div className="ua-curriculum-depth">
      <ConceptComparisonCards comparisons={depth.comparisons} />
      <CaveatBoxes caveats={depth.caveats} />
      <FailureGallery failures={depth.failures} />
      <PaperReadingMode signals={depth.paperSignals} />
    </div>
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
  const [page, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil((terms?.length || 0) / GLOSSARY_PAGE_SIZE));
  const activePage = Math.min(page, pageCount - 1);
  const start = activePage * GLOSSARY_PAGE_SIZE;
  const visibleTerms = terms.slice(start, start + GLOSSARY_PAGE_SIZE);
  const showPagination = terms.length > GLOSSARY_PAGE_SIZE;

  return (
    <section id="math-glossary" className="ua-glossary-panel">
      <div className="ua-glossary-head">
        <span>Glossary</span>
        <h2>Terms in this animation</h2>
        {showPagination && (
          <p>
            Showing {start + 1}-{Math.min(start + GLOSSARY_PAGE_SIZE, terms.length)} of {terms.length} terms
          </p>
        )}
      </div>
      <div className="ua-glossary-grid">
        {visibleTerms.map((term) => (
          <article key={term.id} id={`glossary-${term.slug}`}>
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
      {showPagination && (
        <nav className="ua-glossary-pagination" aria-label="Lesson glossary pagination">
          <button
            type="button"
            onClick={() => setPage((value) => Math.max(0, value - 1))}
            disabled={activePage === 0}
          >
            Previous
          </button>
          <span>
            Page {activePage + 1} of {pageCount}
          </span>
          <button
            type="button"
            onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}
            disabled={activePage === pageCount - 1}
          >
            Next
          </button>
        </nav>
      )}
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
          <span title={model.chips.difficulty}>{model.chips.difficulty}</span>
          <span title={model.chips.minutes}>{model.chips.minutes}</span>
          <span title={model.chips.prereq}>{model.chips.prereq}</span>
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

      <CurriculumDepthPanels depth={model.depth} />

      <Glossary key={animation.id} terms={model.glossary} />
    </div>
  );
}
