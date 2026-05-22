import React from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, BookOpen, Lightbulb, Sigma } from 'lucide-react';
import { getGlossaryTerm, glossaryTerms } from '../data/glossaryRepository.js';

export default function GlossaryPage() {
  const { slug } = useParams();
  const term = getGlossaryTerm(slug);

  if (!term) {
    return (
      <main className="ua-glossary-page">
        <Link to="/" className="ua-back-link">
          <ArrowLeft size={15} />
          Back to index
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
    .slice(0, 6);

  return (
    <main className="ua-glossary-page">
      <Link to="/" className="ua-back-link">
        <ArrowLeft size={15} />
        Back to index
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

      {relatedTerms.length > 0 && (
        <section className="ua-related-terms">
          <div className="ua-glossary-head">
            <span>Related</span>
            <h2>{term.category} terms</h2>
          </div>
          <div>
            {relatedTerms.map((entry) => (
              <Link key={entry.id} to={entry.href}>
                <img src={entry.image.src} alt={entry.image.alt} />
                <span>{entry.term}</span>
              </Link>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
