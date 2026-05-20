// Direction A — Distill.pub style: warm paper, Source Serif headings, generous whitespace,
// hairline rules, single restrained accent. Math set in KaTeX (Computer Modern).

const D = {
  paper: '#fbf8f1',
  ink: '#1a1a1a',
  faint: '#4a4a4a',
  rule: '#d9d2c0',
  accent: '#264273',   // deep ink-blue
  warm: '#a85a3a',     // marginalia red
  curve: '#264273',
  ball: '#a85a3a',
  grid: '#ece6d3',
};

function DistillVariant() {
  window.useKatexReady();
  const K = window.K;
  const history = window.simulate(3.0, 0.25, 12);
  const stepIdx = 4;
  const here = history[stepIdx];
  const grad = 2 * here.w;

  return (
    <div style={{
      width: 1280, height: 920,
      background: D.paper,
      color: D.ink,
      fontFamily: 'Inter, system-ui, sans-serif',
      fontFeatureSettings: '"ss01","cv11"',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <header style={{
        padding: '28px 96px 22px',
        borderBottom: `1px solid ${D.rule}`,
      }}>
        <div style={{
          display: 'flex', alignItems: 'baseline', gap: 16,
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase',
          color: D.faint,
        }}>
          <span>Chapter 04</span>
          <span style={{ color: D.rule }}>·</span>
          <span>Optimization</span>
          <span style={{ marginLeft: 'auto' }}>{history.length - 1} iterations · α = 0.25</span>
        </div>
        <h1 style={{
          fontFamily: '"Source Serif 4", Georgia, serif',
          fontWeight: 400,
          fontSize: 44, lineHeight: 1.05,
          margin: '10px 0 8px',
          letterSpacing: '-0.015em',
        }}>
          Gradient descent on a convex loss
        </h1>
        <p style={{
          fontFamily: '"Source Serif 4", Georgia, serif',
          fontStyle: 'italic',
          fontSize: 16, lineHeight: 1.4,
          margin: 0,
          color: D.faint,
          maxWidth: 760,
        }}>
          A scalar walk-through of the simplest optimizer there is — how a single learning rate <K tex="\alpha" /> bends a parameter <K tex="w" /> toward a minimum, and what changes when we set it too small or too large.
        </p>
      </header>

      {/* Body — two columns: viz + side notes */}
      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '1fr 260px',
        gap: 48,
        padding: '24px 96px 28px',
        minHeight: 0,
      }}>
        {/* Left column */}
        <div>
          {/* Update rule */}
          <div style={{
            background: 'rgba(38,66,115,0.04)',
            border: `1px solid ${D.rule}`,
            padding: '14px 22px',
            marginBottom: 18,
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase',
              color: D.accent, fontWeight: 600,
            }}>Update rule</div>
            <div style={{ fontSize: 19 }}>
              <K tex="w_{t+1} \;=\; w_t \;-\; \alpha \, \nabla_{\!w}\, \mathcal{L}(w_t)" displayMode={false} />
            </div>
            <div style={{
              fontFamily: '"Source Serif 4", serif', fontStyle: 'italic',
              fontSize: 14, color: D.faint,
            }}>with <K tex="\mathcal{L}(w) = w^2" /></div>
          </div>

          {/* Loss landscape figure */}
          <figure style={{ margin: 0 }}>
            <DistillLandscape here={here} grad={grad} />
            <figcaption style={{
              fontFamily: '"Source Serif 4", serif',
              fontStyle: 'italic',
              fontSize: 13, lineHeight: 1.5,
              color: D.faint,
              borderTop: `1px solid ${D.rule}`,
              paddingTop: 10, marginTop: 8,
              maxWidth: 760,
            }}>
              <strong style={{ fontStyle: 'normal', color: D.ink }}>Figure 1.</strong>{' '}
              The loss surface <K tex="\mathcal{L}(w)=w^2" />. The marker shows <K tex={`w_4 = ${here.w.toFixed(3)}`} />; the tangent indicates the gradient <K tex="\nabla\mathcal{L}=2w" /> at that point.
            </figcaption>
          </figure>

          {/* Readouts + history side by side */}
          <div style={{
            display: 'grid', gridTemplateColumns: '300px 1fr', gap: 28,
            marginTop: 20,
          }}>
            <DistillReadouts here={here} grad={grad} stepIdx={stepIdx} />
            <DistillHistory history={history} stepIdx={stepIdx} />
          </div>
        </div>

        {/* Right column — margin notes & controls */}
        <aside>
          <DistillControls />
        </aside>
      </div>
    </div>
  );
}

function DistillLandscape({ here, grad }) {
  // viewBox math: w in [-4.2, 4.2], L in [-1, 18]
  const W = 760, H = 280, PX = 56, PY = 24;
  const xs = (w) => PX + ((w + 4.2) / 8.4) * (W - 2 * PX);
  const ys = (L) => H - PY - ((L + 1) / 19) * (H - 2 * PY);
  const curvePts = [];
  for (let w = -4.2; w <= 4.2; w += 0.05) curvePts.push(`${xs(w).toFixed(2)},${ys(w * w).toFixed(2)}`);
  const ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
  const lTicks = [0, 4, 8, 12, 16];

  // tangent line segment at `here`
  const slope = grad; // dL/dw
  const tx0 = here.w - 1.4, tx1 = here.w + 1.4;
  const ty0 = here.L + slope * (tx0 - here.w);
  const ty1 = here.L + slope * (tx1 - here.w);

  // trail of prior points
  const trail = window.simulate(3.0, 0.25, 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      {/* faint grid */}
      {ticks.map((t) => (
        <line key={`vx${t}`} x1={xs(t)} x2={xs(t)} y1={PY} y2={H - PY}
          stroke={D.grid} strokeWidth="1" />
      ))}
      {lTicks.map((t) => (
        <line key={`hl${t}`} x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)}
          stroke={D.grid} strokeWidth="1" />
      ))}

      {/* axes */}
      <line x1={PX} x2={W - PX} y1={ys(0)} y2={ys(0)} stroke={D.ink} strokeWidth="1" />
      <line x1={xs(0)} x2={xs(0)} y1={PY} y2={H - PY} stroke={D.ink} strokeWidth="1" />

      {/* tick labels */}
      {ticks.map((t) => (
        <text key={`tx${t}`} x={xs(t)} y={ys(0) + 16}
          fontSize="11" fontFamily="JetBrains Mono, monospace"
          textAnchor="middle" fill={D.faint}>{t}</text>
      ))}
      {lTicks.filter((t) => t > 0).map((t) => (
        <text key={`tl${t}`} x={xs(0) - 8} y={ys(t) + 4}
          fontSize="11" fontFamily="JetBrains Mono, monospace"
          textAnchor="end" fill={D.faint}>{t}</text>
      ))}

      {/* axis labels */}
      <text x={W - PX + 6} y={ys(0) + 4} fontSize="13" fontStyle="italic"
        fontFamily='"Source Serif 4", serif' fill={D.ink}>w</text>
      <text x={xs(0) + 8} y={PY + 4} fontSize="13" fontStyle="italic"
        fontFamily='"Source Serif 4", serif' fill={D.ink}>ℒ(w)</text>

      {/* curve */}
      <polyline points={curvePts.join(' ')} fill="none"
        stroke={D.curve} strokeWidth="1.6" />

      {/* tangent line */}
      <line x1={xs(tx0)} y1={ys(ty0)} x2={xs(tx1)} y2={ys(ty1)}
        stroke={D.warm} strokeWidth="1.25" strokeDasharray="4 4" />

      {/* trail of prior steps */}
      {trail.slice(0, -1).map((p, i) => (
        <g key={`tr${i}`} opacity={0.35 + (i / trail.length) * 0.5}>
          <circle cx={xs(p.w)} cy={ys(p.L)} r="3" fill={D.ball} />
        </g>
      ))}

      {/* current point */}
      <circle cx={xs(here.w)} cy={ys(here.L)} r="6.5" fill={D.paper} stroke={D.ball} strokeWidth="2" />
      <circle cx={xs(here.w)} cy={ys(here.L)} r="3" fill={D.ball} />

      {/* annotation */}
      <g>
        <line x1={xs(here.w) + 10} y1={ys(here.L) - 14}
          x2={xs(here.w) + 64} y2={ys(here.L) - 44}
          stroke={D.faint} strokeWidth="0.8" />
        <text x={xs(here.w) + 70} y={ys(here.L) - 42}
          fontSize="12" fontStyle="italic"
          fontFamily='"Source Serif 4", serif' fill={D.faint}>
          step 4 · w = {here.w.toFixed(3)}
        </text>
      </g>
    </svg>
  );
}

function DistillReadouts({ here, grad }) {
  const K = window.K;
  const rows = [
    { label: 'iteration', tex: 't', val: '4' },
    { label: 'weight',    tex: 'w_t', val: here.w.toFixed(3) },
    { label: 'gradient',  tex: '\\nabla\\mathcal{L}', val: grad.toFixed(3) },
    { label: 'loss',      tex: '\\mathcal{L}(w_t)', val: here.L.toFixed(3) },
  ];
  return (
    <div style={{ borderTop: `1px solid ${D.ink}`, borderBottom: `1px solid ${D.rule}` }}>
      {rows.map((r, i) => (
        <div key={r.label} style={{
          display: 'grid', gridTemplateColumns: '1fr 1fr auto',
          alignItems: 'baseline',
          padding: '9px 4px',
          borderBottom: i < rows.length - 1 ? `1px solid ${D.rule}` : 'none',
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase',
            color: D.faint,
          }}>{r.label}</div>
          <div style={{ fontSize: 16, color: D.faint }}><K tex={r.tex} /></div>
          <div style={{
            fontFamily: '"Source Serif 4", serif',
            fontSize: 20, fontVariantNumeric: 'tabular-nums',
            color: D.ink,
          }}>{r.val}</div>
        </div>
      ))}
    </div>
  );
}

function DistillHistory({ history, stepIdx }) {
  const W = 360, H = 168, PX = 36, PY = 16;
  const maxL = 9;
  const xs = (i) => PX + (i / (history.length - 1)) * (W - 2 * PX);
  const ys = (L) => H - PY - (L / maxL) * (H - 2 * PY);
  const pts = history.map((p) => `${xs(p.i).toFixed(2)},${ys(p.L).toFixed(2)}`).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase',
        color: D.accent, marginBottom: 6,
      }}>Loss history</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
        {[0, 3, 6, 9].map((t) => (
          <g key={t}>
            <line x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)} stroke={D.grid} />
            <text x={PX - 6} y={ys(t) + 4} fontSize="10" textAnchor="end"
              fontFamily="JetBrains Mono, monospace" fill={D.faint}>{t}</text>
          </g>
        ))}
        <line x1={PX} x2={W - PX} y1={H - PY} y2={H - PY} stroke={D.ink} />
        <line x1={PX} x2={PX} y1={PY} y2={H - PY} stroke={D.ink} />
        <polyline points={pts} fill="none" stroke={D.curve} strokeWidth="1.6" />
        {history.map((p, i) => (
          <circle key={i} cx={xs(p.i)} cy={ys(p.L)} r={i === stepIdx ? 4 : 2.2}
            fill={i === stepIdx ? D.ball : D.curve}
            stroke={i === stepIdx ? D.paper : 'none'} strokeWidth="1.5" />
        ))}
        <text x={W - PX} y={H - 4} fontSize="11" textAnchor="end"
          fontStyle="italic" fontFamily='"Source Serif 4", serif' fill={D.faint}>iteration t</text>
      </svg>
      <figcaption style={{
        fontFamily: '"Source Serif 4", serif',
        fontStyle: 'italic', fontSize: 12, color: D.faint,
        borderTop: `1px solid ${D.rule}`, paddingTop: 6, marginTop: 4,
      }}>
        <strong style={{ fontStyle: 'normal', color: D.ink }}>Figure 2.</strong>{' '}
        ℒ decays geometrically when α &lt; 1.
      </figcaption>
    </figure>
  );
}

function DistillControls() {
  const K = window.K;
  return (
    <div style={{
      fontFamily: '"Source Serif 4", serif',
      fontSize: 14, lineHeight: 1.55, color: D.faint,
    }}>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase',
        color: D.accent, marginBottom: 12, fontWeight: 600,
      }}>Controls</div>

      <DistillSlider label="Learning rate" tex="\alpha" value="0.25" pos={0.25} hint="α &lt; 1: contraction" />
      <DistillSlider label="Initial weight" tex="w_0" value="+3.00" pos={0.8} hint="symmetric about origin" />

      <p style={{ margin: '24px 0 0', fontStyle: 'italic' }}>
        Try <K tex="\alpha = 0.01" /> — convergence is monotone but slow. At <K tex="\alpha = 0.95" /> the iterates oscillate; past <K tex="\alpha = 1" /> they diverge.
      </p>

      <div style={{
        marginTop: 28, display: 'flex', gap: 10,
      }}>
        <button style={{
          flex: 1,
          background: D.ink, color: D.paper,
          border: 'none', padding: '12px 16px',
          fontFamily: 'inherit', fontSize: 14, letterSpacing: '0.02em',
          cursor: 'pointer',
        }}>Run descent →</button>
        <button style={{
          background: 'transparent', color: D.ink,
          border: `1px solid ${D.ink}`, padding: '12px 14px',
          fontFamily: 'inherit', fontSize: 14,
          cursor: 'pointer',
        }}>Reset</button>
      </div>
    </div>
  );
}

function DistillSlider({ label, tex, value, pos, hint }) {
  const K = window.K;
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{
        display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <div>
          <span style={{ color: D.ink, fontFamily: 'Inter, sans-serif', fontSize: 13 }}>{label} </span>
          <span style={{ fontSize: 15 }}><K tex={tex} /></span>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 13, color: D.ink, fontVariantNumeric: 'tabular-nums',
        }}>{value}</div>
      </div>
      <div style={{ position: 'relative', height: 1, background: D.ink, marginBottom: 6 }}>
        <div style={{
          position: 'absolute', left: `${pos * 100}%`,
          top: -5, width: 10, height: 10,
          background: D.paper, border: `1.5px solid ${D.ink}`,
          transform: 'translateX(-50%) rotate(45deg)',
        }} />
      </div>
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 10, letterSpacing: '0.08em',
        color: D.warm, fontStyle: 'normal',
      }}>{hint}</div>
    </div>
  );
}

window.DistillVariant = DistillVariant;
