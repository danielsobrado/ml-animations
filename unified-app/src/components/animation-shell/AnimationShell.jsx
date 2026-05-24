import React, { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import { AlertTriangle, BookOpen, ClipboardCheck, FileSearch, GitCompare, Library, ShieldCheck } from 'lucide-react';
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

function LessonLinks({ lessonIds }) {
  if (!lessonIds?.length) return null;

  const lessons = lessonIds
    .map((lessonId) => allAnimations.find((animation) => animation.id === lessonId))
    .filter(Boolean)
    .slice(0, 4);

  if (!lessons.length) return null;

  return (
    <div>
      <h4>Lesson links</h4>
      <div className="ua-depth-link-list">
        {lessons.map((lesson) => (
          <Link key={lesson.id} to={`/animation/${lesson.id}`}>{lesson.name}</Link>
        ))}
      </div>
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
            {comparison.whenToUseLeft && (
              <p><b>Use {comparison.left} when:</b> {comparison.whenToUseLeft}</p>
            )}
            {comparison.whenToUseRight && (
              <p><b>Use {comparison.right} when:</b> {comparison.whenToUseRight}</p>
            )}
            <p><b>Common mistake:</b> {comparison.commonMistake}</p>
            {comparison.failureIfConfused && (
              <p><b>If confused:</b> {comparison.failureIfConfused}</p>
            )}
            <p><b>Diagnostic:</b> {comparison.diagnostic}</p>
            {comparison.tinyScenario && (
              <p><b>Tiny scenario:</b> {comparison.tinyScenario}</p>
            )}
            <LessonLinks lessonIds={comparison.lessonLinks || comparison.lessonIds} />
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
            {(failure.severity || failure.learnerLevel) && (
              <p><b>Signal:</b> {[failure.severity, failure.learnerLevel].filter(Boolean).join(' / ')}</p>
            )}
            {failure.minimalScenario && <p><b>Minimal scenario:</b> {failure.minimalScenario}</p>}
            <p><b>Symptom:</b> {failure.symptom}</p>
            <p><b>Why:</b> {failure.whyItHappens}</p>
            <p><b>Detect:</b> {failure.howToDetect}</p>
            <p><b>Fix:</b> {failure.howToFix}</p>
            {failure.falseFix && <p><b>False fix:</b> {failure.falseFix}</p>}
            {failure.tryInLesson?.lessonId && (
              <p>
                <b>Try in lesson:</b>{' '}
                <Link to={`/animation/${failure.tryInLesson.lessonId}`}>
                  {failure.tryInLesson.control || failure.tryInLesson.lessonId}
                </Link>
              </p>
            )}
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
            {(signal.sourceType || signal.sourceName || signal.sourceYear) && (
              <p>
                <b>Provenance:</b>{' '}
                {[signal.sourceType, signal.sourceName, signal.sourceYear].filter(Boolean).join(' / ')}
              </p>
            )}
            {(signal.claimStatus || signal.freshnessDate || signal.confidence) && (
              <p>
                <b>Claim status:</b>{' '}
                {[signal.claimStatus, signal.freshnessDate, signal.confidence].filter(Boolean).join(' / ')}
              </p>
            )}
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

function ToyModelNotes({ caveat }) {
  const toyModel = caveat.toyModel || null;
  const whatIsSimplified = toyModel?.whatIsSimplified || caveat.whatIsSimplified || caveat.proxyMetrics;
  const whatStillHolds = toyModel?.whatStillHolds || caveat.whatStillHolds;
  const productionReality = toyModel?.whatWouldChangeInProduction || caveat.productionReality;
  const toyFormula = toyModel?.toyFormula || caveat.toyFormula;

  if (!toyFormula && !whatIsSimplified?.length && !whatStillHolds?.length && !productionReality?.length) {
    return null;
  }

  return (
    <div className="ua-depth-subsection">
      <h4>Teaching simplification</h4>
      {toyFormula && <p><b>Toy formula:</b> {toyFormula}</p>}
      <DepthList title="What is simplified" items={whatIsSimplified} />
      <DepthList title="What still holds" items={whatStillHolds} />
      <DepthList title="Production reality" items={productionReality} />
    </div>
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
            <ToyModelNotes caveat={caveat} />
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

function MathControls({ model, onReset, onFocusStage, onOpenGlossary }) {
  const [prereq] = model.mindmap.prereqs;
  const [next] = model.mindmap.next;

  const actions = {
    prereq: prereq ? { as: Link, to: `/animation/${prereq.id}` } : { as: 'button', onClick: onFocusStage },
    reset: { as: 'button', onClick: onReset },
    play: { as: 'button', onClick: onFocusStage },
    sum: { as: 'button', onClick: onOpenGlossary },
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

function getLessonWorkspaceTabs({
  assessment,
  depth,
  glossaryTerms,
  lessonId,
}) {
  const hasDepth = depth && (
    depth.comparisons.length > 0
    || depth.failures.length > 0
    || depth.paperSignals.length > 0
    || depth.caveats.length > 0
  );

  return [
    assessment && {
      id: 'check',
      label: 'Core questions',
      icon: ClipboardCheck,
      panel: <AssessmentPanel lessonId={lessonId} eyebrow="Progress" title="Core questions" />,
    },
    glossaryTerms?.length > 0 && {
      id: 'glossary',
      label: 'Glossary',
      icon: Library,
      panel: <Glossary key={lessonId} terms={glossaryTerms} />,
    },
    hasDepth && {
      id: 'deep-dive',
      label: 'Deep dive',
      icon: FileSearch,
      panel: <CurriculumDepthPanels depth={depth} />,
    },
  ].filter(Boolean);
}

function WorkspaceTabBar({ activeTab, tabs, onTabChange, className = '' }) {
  if (!tabs.length) return null;

  return (
    <div className={`ua-workspace-tabs ${className}`.trim()} aria-label="Lesson workspace sections" role="tablist">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const selected = tab.id === activeTab;

        return (
          <button
            type="button"
            key={tab.id}
            className="ua-workspace-tab"
            role="tab"
            aria-selected={selected}
            aria-controls={`workspace-panel-${tab.id}`}
            id={`workspace-tab-${tab.id}`}
            onClick={() => onTabChange(tab.id)}
          >
            <Icon size={15} />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

function findLessonToolbar(stage) {
  const nav = stage.querySelector('.ua-stage-wrap nav:not(.ds-tabs)');
  if (!nav) return null;

  return nav.querySelector(':scope > div > div')
    || nav.querySelector(':scope > div')
    || nav;
}

function WorkspaceTabPortal({ activeTab, tabs, onTabChange }) {
  const [portalSlot, setPortalSlot] = useState(null);

  useEffect(() => {
    const stage = document.getElementById('math-main-stage');
    if (!stage || tabs.length === 0) return undefined;

    let slot = null;

    const attach = () => {
      const toolbar = findLessonToolbar(stage);
      if (!toolbar || slot?.isConnected) return Boolean(slot?.isConnected);

      slot = document.createElement('div');
      slot.className = 'ua-workspace-portal-slot';
      toolbar.appendChild(slot);
      setPortalSlot(slot);
      return true;
    };

    if (attach()) {
      return () => {
        slot?.remove();
        setPortalSlot(null);
      };
    }

    const observer = new MutationObserver(() => {
      if (attach()) observer.disconnect();
    });
    observer.observe(stage, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      slot?.remove();
      setPortalSlot(null);
    };
  }, [tabs.length]);

  if (!portalSlot) {
    return (
      <WorkspaceTabBar
        activeTab={activeTab}
        tabs={tabs}
        onTabChange={onTabChange}
        className="ua-workspace-tabs-inline"
      />
    );
  }

  return createPortal(
    <WorkspaceTabBar
      activeTab={activeTab}
      tabs={tabs}
      onTabChange={onTabChange}
      className="ua-workspace-tabs-inline"
    />,
    portalSlot,
  );
}

function LessonWorkspace({
  activeTab,
  tabs,
}) {
  if (tabs.length === 0) return null;

  const selectedTab = tabs.some((tab) => tab.id === activeTab) ? activeTab : null;
  if (!selectedTab) return null;

  const selectedPanel = tabs.find((tab) => tab.id === selectedTab)?.panel;

  return (
    <section id="lesson-workspace" className="ua-lesson-workspace" aria-label="Lesson workspace">
      <div
        className="ua-workspace-panel"
        id={`workspace-panel-${selectedTab}`}
        role="tabpanel"
        aria-labelledby={`workspace-tab-${selectedTab}`}
      >
        {selectedPanel}
      </div>
    </section>
  );
}

export default function AnimationShell({ animation, children }) {
  const [resetNonce, setResetNonce] = useState(0);
  const [workspaceTab, setWorkspaceTab] = useState('lesson');
  const model = useMemo(() => createLearningModel(animation, allAnimations), [animation]);
  const assessment = useMemo(() => getLessonAssessment(animation.id), [animation.id]);
  const showShellAssessment = hasAssessmentContent(assessment);
  const workspaceTabs = useMemo(() => getLessonWorkspaceTabs({
    assessment: showShellAssessment,
    depth: model.depth,
    glossaryTerms: model.glossary,
    lessonId: animation.id,
  }), [animation.id, model.depth, model.glossary, showShellAssessment]);

  const hasActiveWorkspaceTab = workspaceTabs.some((tab) => tab.id === workspaceTab);

  useEffect(() => {
    setWorkspaceTab('lesson');
  }, [animation.id]);

  useEffect(() => {
    const stage = document.getElementById('math-main-stage');
    if (!stage) return undefined;

    const clearWorkspaceTab = (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('.ua-workspace-portal-slot')) return;
      const button = target.closest('button');
      if (button?.closest('.ua-stage-wrap > :first-child > nav:not(.ds-tabs)')) {
        setWorkspaceTab('lesson');
      }
    };

    stage.addEventListener('click', clearWorkspaceTab, true);
    return () => stage.removeEventListener('click', clearWorkspaceTab, true);
  }, [animation.id]);

  const resetStage = () => {
    setResetNonce((value) => value + 1);
    document.getElementById('math-main-stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const focusStage = () => {
    document.getElementById('math-main-stage')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const openWorkspaceTab = (tabId) => {
    if (!workspaceTabs.some((tab) => tab.id === tabId)) return;
    setWorkspaceTab(tabId);
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

      <MathControls
        model={model}
        onReset={resetStage}
        onFocusStage={focusStage}
        onOpenGlossary={() => openWorkspaceTab('glossary')}
      />

      <Suspense fallback={<MindmapFallback />}>
        <ConceptMindmap mindmap={model.mindmap} />
      </Suspense>

      <div className="ua-learning-grid">
        <main id="math-main-stage" className="ua-main-stage" aria-label={`${animation.name} animation stage`}>
          <WorkspaceTabPortal
            activeTab={workspaceTab}
            tabs={workspaceTabs}
            onTabChange={openWorkspaceTab}
          />
          <div
            key={resetNonce}
            className={[
              'ua-stage-wrap',
              showShellAssessment && 'has-shell-assessment',
              hasActiveWorkspaceTab && 'has-workspace-tab',
            ].filter(Boolean).join(' ')}
          >
            {children}
            <LessonWorkspace
              activeTab={workspaceTab}
              tabs={workspaceTabs}
            />
          </div>
        </main>

        <LearningCards cards={model.learningCards} />
      </div>
    </div>
  );
}
