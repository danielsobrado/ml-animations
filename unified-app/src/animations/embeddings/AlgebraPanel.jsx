import React, { useState } from 'react';
import { motion } from 'framer-motion';

const steps = [
  '1. Start: King',
  '2. Subtract: Man',
  '3. Add: Woman',
];

export default function AlgebraPanel() {
  const king = { x: 100, y: 300 };
  const man = { x: 50, y: 100 };
  const woman = { x: 250, y: 100 };
  const queen = { x: 300, y: 300 };
  const [step, setStep] = useState(0);

  const getResultVector = () => {
    let x = king.x;
    let y = king.y;

    if (step >= 1) {
      x -= man.x;
      y -= man.y;
    }
    if (step >= 2) {
      x += woman.x;
      y += woman.y;
    }
    return { x, y };
  };

  const result = getResultVector();

  return (
    <div className="embeddings-algebra-panel">
      <div className="embeddings-algebra-intro">
        <h2>Word Algebra</h2>
        <p>
          Embeddings capture meaning as direction.
          <span>King - Man + Woman &asymp; Queen</span>
        </p>
      </div>

      <div className="embeddings-algebra-steps" role="group" aria-label="Word algebra steps">
        {steps.map((label, index) => (
          <button
            key={label}
            type="button"
            onClick={() => setStep(index)}
            className={step === index ? 'is-active' : ''}
            aria-pressed={step === index}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="embeddings-algebra-figure">
        <svg className="w-full h-full" viewBox="0 0 600 400" role="img" aria-label="Vector arithmetic from king minus man plus woman toward queen">
          <defs>
            <marker id="embeddings-arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--ds-mute)" />
            </marker>
            <marker id="embeddings-arrow-active" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--ds-accent)" />
            </marker>
            <marker id="embeddings-arrow-subtract" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#9b5f2b" />
            </marker>
            <pattern id="embeddings-grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="var(--ds-grid)" strokeWidth="0.7" />
            </pattern>
          </defs>

          <rect width="100%" height="100%" fill="url(#embeddings-grid)" />

          <line x1="0" y1="400" x2={king.x} y2={400 - king.y} stroke="var(--ds-mute)" strokeWidth="1.8" strokeDasharray="5 5" markerEnd="url(#embeddings-arrow)" />
          <text x={king.x} y={400 - king.y - 10} fill="var(--ds-mute)" textAnchor="middle" fontSize="12">King</text>

          <line x1="0" y1="400" x2={man.x} y2={400 - man.y} stroke="var(--ds-mute)" strokeWidth="1.8" strokeDasharray="5 5" markerEnd="url(#embeddings-arrow)" />
          <text x={man.x} y={400 - man.y - 10} fill="var(--ds-mute)" textAnchor="middle" fontSize="12">Man</text>

          <line x1="0" y1="400" x2={woman.x} y2={400 - woman.y} stroke="var(--ds-mute)" strokeWidth="1.8" strokeDasharray="5 5" markerEnd="url(#embeddings-arrow)" />
          <text x={woman.x} y={400 - woman.y - 10} fill="var(--ds-mute)" textAnchor="middle" fontSize="12">Woman</text>

          <circle cx={queen.x} cy={400 - queen.y} r="5" fill="var(--ds-accent)" />
          <text x={queen.x} y={400 - queen.y - 15} fill="var(--ds-accent)" textAnchor="middle" fontWeight="700">Queen (Target)</text>

          <motion.g initial={false} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
            <line
              x1="0"
              y1="400"
              x2={king.x}
              y2={400 - king.y}
              stroke="var(--ds-accent)"
              strokeWidth="3.5"
              markerEnd="url(#embeddings-arrow-active)"
            />

            {step >= 1 && (
              <line
                x1={king.x}
                y1={400 - king.y}
                x2={king.x - man.x}
                y2={400 - king.y + man.y}
                stroke="#9b5f2b"
                strokeWidth="3.5"
                markerEnd="url(#embeddings-arrow-subtract)"
              />
            )}

            {step >= 2 && (
              <line
                x1={king.x - man.x}
                y1={400 - king.y + man.y}
                x2={result.x}
                y2={400 - result.y}
                stroke="var(--ds-accent)"
                strokeWidth="3.5"
                markerEnd="url(#embeddings-arrow-active)"
              />
            )}
          </motion.g>

          <motion.circle
            animate={{ cx: result.x, cy: 400 - result.y }}
            r="8"
            fill="#b7791f"
            stroke="var(--ds-paper)"
            strokeWidth="2"
          />
          <motion.text
            animate={{ x: result.x, y: 400 - result.y + 25 }}
            fill="#8a4d12"
            textAnchor="middle"
            fontWeight="700"
          >
            Result
          </motion.text>
        </svg>
      </div>
    </div>
  );
}
