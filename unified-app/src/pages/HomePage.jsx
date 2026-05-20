import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { allAnimations, categories, curriculumBacklog, curriculumTracks } from '../data/animations';

export default function HomePage() {
  const totalAnimations = categories.reduce((sum, category) => sum + category.items.length, 0);
  const animationById = new Map(allAnimations.map((animation) => [animation.id, animation]));
  const backlogByTrack = curriculumBacklog.reduce((acc, topic) => {
    acc[topic.trackId] = [...(acc[topic.trackId] || []), topic];
    return acc;
  }, {});

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
          neural networks, transformers, diffusion, reinforcement learning, and linear algebra,
          now arranged into guided learning tracks with prerequisites and next steps.
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
          <strong>React</strong>
          <span>Vite apps</span>
        </div>
        <div>
          <strong>KaTeX</strong>
          <span>Math notes</span>
        </div>
      </section>

      <section className="ua-tracks" aria-labelledby="guided-tracks-title">
        <div className="ua-section-head">
          <span>Guided paths</span>
          <h2 id="guided-tracks-title">Curriculum tracks</h2>
          <p>Follow these paths when you want sequencing; use the catalog below when you want reference browsing.</p>
        </div>

        <div className="ua-track-grid">
          {curriculumTracks.map((track, trackIndex) => {
            const firstAnimation = animationById.get(track.animationIds[0]);
            const activeMinutes = track.animationIds.reduce(
              (sum, id) => sum + (animationById.get(id)?.estimatedMinutes || 0),
              0,
            );
            const plannedTopics = backlogByTrack[track.id] || [];

            return (
              <article className="ua-track-card" key={track.id}>
                <div className="ua-track-card-head">
                  <span>{String(trackIndex + 1).padStart(2, '0')}</span>
                  <strong>{track.title}</strong>
                </div>
                <p>{track.description}</p>
                <div className="ua-track-meta">
                  <span>{track.animationIds.length} active</span>
                  <span>{activeMinutes} min</span>
                  <span>{plannedTopics.length} planned</span>
                </div>
                <div className="ua-track-sequence">
                  {track.animationIds.slice(0, 4).map((animationId) => {
                    const animation = animationById.get(animationId);
                    if (!animation) return null;
                    return <span key={animation.id}>{animation.name}</span>;
                  })}
                </div>
                {plannedTopics.length > 0 && (
                  <div className="ua-track-planned">
                    Planned: {plannedTopics.slice(0, 2).map((topic) => topic.title).join('; ')}
                  </div>
                )}
                {firstAnimation && (
                  <Link className="ua-track-link" to={`/animation/${firstAnimation.id}`}>
                    Start track <ArrowRight size={16} />
                  </Link>
                )}
              </article>
            );
          })}
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
