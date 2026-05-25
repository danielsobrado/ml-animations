import React from 'react';
import { Link } from 'react-router-dom';

function normalizeTooltip(tooltip) {
  if (!tooltip) return null;
  if (typeof tooltip === 'string') {
    return { short: tooltip };
  }
  return tooltip;
}

function TooltipSection({ title, value, children }) {
  const content = value ?? children;
  if (!content) return null;
  return (
    <div className="ua-map-tooltip-section">
      <h4>{title}</h4>
      <div className="ua-map-tooltip-body">{typeof content === 'string' ? <p>{content}</p> : content}</div>
    </div>
  );
}

export default function ConceptTooltip({ selection }) {
  const tooltip = normalizeTooltip(selection?.tooltip);

  if (!selection || !tooltip) {
    return (
      <aside className="ua-map-detail" aria-label="Concept details">
        <p className="ua-map-detail-placeholder">
          Select a concept in the map to see meaning, intuition, examples, and traps.
        </p>
      </aside>
    );
  }

  return (
    <aside className="ua-map-detail" aria-label="Concept details">
      <header className="ua-map-detail-head">
        <h3>{selection.label}</h3>
        {selection.lessonId ? (
          <Link className="ua-map-detail-link" to={`/animation/${selection.lessonId}`}>
            Open lesson
          </Link>
        ) : null}
      </header>

      <TooltipSection title="Meaning" value={tooltip.short} />
      <TooltipSection title="Intuition" value={tooltip.intuition} />
      <TooltipSection
        title="Formula"
        value={tooltip.formula ? <p className="ua-map-detail-formula">{tooltip.formula}</p> : null}
      />
      <TooltipSection
        title="Code"
        value={tooltip.code ? <pre className="ua-map-detail-code">{tooltip.code}</pre> : null}
      />
      <TooltipSection title="Example" value={tooltip.example} />
      <TooltipSection title="Watch out" value={tooltip.trap} />
      <TooltipSection title="Used later" value={tooltip.why} />
    </aside>
  );
}
