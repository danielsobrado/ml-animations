import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { categories } from '../data/animations';

const totalLabs = 41;
const totalQuizQuestions = 161;

export default function HomePage() {
  const totalAnimations = categories.reduce((sum, category) => sum + category.items.length, 0);

  return (
    <div className="ua-home">
      <section className="ua-home-hero">
        <div className="ua-home-eyebrow">
          <span>Catalog</span>
          <span className="sep">/</span>
          <span>{totalAnimations} interactive notes</span>
        </div>
        <h1 className="ua-home-title">
          Machine learning <span className="accent">visualized</span>.
        </h1>
        <p className="ua-home-subtitle">
          A browsable set of animated explanations for optimization, probability,
          neural networks, transformers, diffusion, reinforcement learning, and linear algebra.
        </p>
      </section>

      <section className="ua-home-stats" aria-label="Catalog statistics">
        <div>
          <strong>{categories.length}</strong>
          <span>Chapters</span>
        </div>
        <div>
          <strong>{totalAnimations}</strong>
          <span>Animations</span>
        </div>
        <div>
          <strong>{totalLabs}</strong>
          <span>Practice labs</span>
        </div>
        <div>
          <strong>{totalQuizQuestions}</strong>
          <span>Quiz questions</span>
        </div>
      </section>

      <div className="ua-toc">
        {categories.map((category, categoryIndex) => (
          <section className="ua-toc-section" key={category.id}>
            <div className="ua-toc-head">
              <span>{String(categoryIndex + 1).padStart(2, '0')}</span>
              <h2>{category.name}</h2>
              <small>{category.items.length} entries</small>
            </div>
            <div className="ua-toc-list">
              {category.items.map((item, itemIndex) => (
                <Link className="ua-toc-item" key={item.id} to={`/animation/${item.id}`}>
                  <span className="ua-toc-num">
                    {String(categoryIndex + 1).padStart(2, '0')}.
                    {String(itemIndex + 1).padStart(2, '0')}
                  </span>
                  <span className="ua-toc-title">{item.name}</span>
                  <span className="ua-toc-desc">{item.description}</span>
                  <ArrowRight size={16} />
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
