import React from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, BookOpen, GitBranch, Lightbulb, Sigma } from 'lucide-react';
import { allAnimations } from '../data/animations.js';
import { getGlossaryTerm, glossaryTerms } from '../data/glossaryRepository.js';

const lessonsById = new Map(allAnimations.map((animation) => [animation.id, animation]));

function labelFromId(id) {
  return String(id)
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function lessonReference(id) {
  const lesson = lessonsById.get(id);
  if (lesson) {
    return {
      id: `lesson:${lesson.id}`,
      kind: 'Lesson',
      label: lesson.name,
      description: lesson.description,
      href: `/animation/${lesson.id}`,
      image: null,
    };
  }

  return null;
}

function termReference(id) {
  const term = getGlossaryTerm(id);
  if (term) {
    return {
      id: `term:${term.id}`,
      kind: term.category,
      label: term.term,
      description: term.definition,
      href: term.href,
      image: term.image,
    };
  }

  return null;
}

function resolveReference(id, prefer = 'term') {
  const resolvers = prefer === 'lesson'
    ? [lessonReference, termReference]
    : [termReference, lessonReference];

  for (const resolver of resolvers) {
    const reference = resolver(id);
    if (reference) return reference;
  }

  return null;
}

function uniqueReferences(ids, excludeIds = [], prefer) {
  const seen = new Set();
  const excluded = new Set(excludeIds);
  return ids
    .map((id) => resolveReference(id, prefer))
    .filter(Boolean)
    .filter((entry) => {
      if (excluded.has(entry.id)) return false;
      if (seen.has(entry.id)) return false;
      seen.add(entry.id);
      return true;
    });
}

function RelationshipSection({ title, eyebrow, ids, fallback, excludeIds, prefer }) {
  const items = uniqueReferences(ids, excludeIds, prefer);
  const fallbackItems = items.length > 0 ? [] : uniqueReferences(fallback || [], excludeIds, prefer);
  const visibleItems = items.length > 0 ? items : fallbackItems;

  if (visibleItems.length === 0) return null;

  return (
    <section className="ua-relationship-section">
      <div className="ua-glossary-head">
        <span>{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      <div className="ua-relationship-grid">
        {visibleItems.map((entry) => (
          <Link key={entry.id} to={entry.href}>
            {entry.image ? (
              <img src={entry.image.src} alt={entry.image.alt} />
            ) : (
              <span className="ua-lesson-ref-icon">
                <BookOpen size={18} />
              </span>
            )}
            <span>{entry.kind}</span>
            <strong>{entry.label}</strong>
            <p>{entry.description || labelFromId(entry.id)}</p>
          </Link>
        ))}
      </div>
    </section>
  );
}

function TermMetadata({ term }) {
  const aliasLabel = term.aliases.length > 0 ? term.aliases.join(', ') : 'none';
  const showSymbol = Boolean(term.symbol && term.symbol !== term.term.slice(0, 1));

  return (
    <dl className="ua-term-meta" aria-label={`${term.term} metadata`}>
      {showSymbol && (
        <div>
          <dt>Symbol</dt>
          <dd>{term.symbol}</dd>
        </div>
      )}
      <div>
        <dt>Aliases</dt>
        <dd>{aliasLabel}</dd>
      </div>
      <div>
        <dt>Slug</dt>
        <dd>{term.slug}</dd>
      </div>
    </dl>
  );
}

export default function GlossaryPage() {
  const { slug } = useParams();
  const term = getGlossaryTerm(slug);

  if (!term) {
    return (
      <main className="ua-glossary-page">
        <Link to="/glossary" className="ua-back-link">
          <ArrowLeft size={15} />
          Back to glossary
        </Link>
        <section className="ds-header">
          <div className="ds-eyebrow">Glossary</div>
          <h1 className="ds-title">Term not found</h1>
          <p className="ds-subtitle">The glossary entry "{slug}" is not in the central repository yet.</p>
        </section>
      </main>
    );
  }

  const relatedTerms = glossaryTerms
    .filter((entry) => entry.category === term.category && entry.id !== term.id)
    .map((entry) => entry.id)
    .slice(0, 6);

  return (
    <main className="ua-glossary-page">
      <Link to="/glossary" className="ua-back-link">
        <ArrowLeft size={15} />
        Back to glossary
      </Link>

      <section className="ua-term-hero">
        <div>
          <div className="ds-eyebrow">
            <span>Glossary</span>
            <span className="sep">/</span>
            <span>{term.category}</span>
          </div>
          <h1>{term.term}</h1>
          <p>{term.definition}</p>
          <TermMetadata term={term} />
        </div>
        <img src={term.image.src} alt={term.image.alt} />
      </section>

      <section className="ua-term-notes">
        <article className="wide">
          <BookOpen size={17} />
          <h2>What it means</h2>
          <p>{term.explanation}</p>
        </article>
        <article>
          <Lightbulb size={17} />
          <h2>Intuition</h2>
          <p>{term.intuition}</p>
        </article>
        <article>
          <Sigma size={17} />
          <h2>Example</h2>
          <p>{term.example}</p>
        </article>
        <article>
          <AlertTriangle size={17} />
          <h2>Common pitfall</h2>
          <p>{term.pitfall}</p>
        </article>
      </section>

      <section className="ua-concept-graph" aria-label={`${term.term} concept graph`}>
        <div className="ua-concept-graph-head">
          <GitBranch size={18} />
          <span>Concept graph</span>
        </div>
        <RelationshipSection
          eyebrow="Lessons"
          title="Used in these lessons"
          ids={term.usedIn}
          excludeIds={[`term:${term.id}`]}
          prefer="lesson"
        />
        <RelationshipSection
          eyebrow="Next concepts"
          title="Prerequisite for"
          ids={term.prerequisiteFor}
          excludeIds={[`term:${term.id}`]}
          prefer="lesson"
        />
        <RelationshipSection
          eyebrow="Watch"
          title="Commonly confused with"
          ids={term.confusedWith}
          excludeIds={[`term:${term.id}`]}
        />
        <RelationshipSection
          eyebrow="See also"
          title="Related concepts"
          ids={term.related}
          fallback={relatedTerms}
          excludeIds={[`term:${term.id}`]}
        />
      </section>
    </main>
  );
}
