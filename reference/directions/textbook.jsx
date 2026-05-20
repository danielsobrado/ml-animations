// Direction C — Modern textbook: white canvas, very sharp typography,
// variables colour-coded across body copy and figures, "definition" callouts.

const T = {
  bg: '#ffffff',
  panel: '#fafaf7',
  ink: '#0c0c0c',
  faint: '#6a6a6a',
  rule: '#e5e3dc',
  hair: '#d6d3c8',

  w: '#1f4ae2',       // weight — cobalt
  alpha: '#c2410c',   // learning rate — burnt orange
  grad: '#7c3aed',    // gradient — violet
  loss: '#0d7a6b',    // loss — teal
};

function TextbookVariant() {
  window.useKatexReady();
  const K = window.K;
  const history = window.simulate(3.0, 0.25, 12);
  const stepIdx = 4;
  const here = history[stepIdx];
  const grad = 2 * here.w;

  return (
    <div style={{
      width: 1280, height: 920,
      background: T.bg,
      color: T.ink,
      fontFamily: 'Inter, system-ui, sans-serif',
      fontFeatureSettings: '"ss01","cv11","tnum"',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Top strip */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '20px 56px',
        borderBottom: `1px solid ${T.rule}`,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: T.faint,
        }}>
          <div style={{
            width: 18, height: 18,
            background: T.ink,
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            color: T.bg, fontFamily: '"Source Serif 4", serif',
            fontStyle: 'italic', fontSize: 13, fontWeight: 500,
          }}>ƒ</div>
          <span style={{ color: T.ink, fontWeight: 600, letterSpacing: '0.08em' }}>ML / Animations</span>
          <span style={{ color: T.hair }}>/</span>
          <span>04 · Optimization</span>
          <span style={{ color: T.hair }}>/</span>
          <span>4.1 Gradient descent</span>
        </div>
        <div style={{
          display: 'flex', gap: 6,
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          color: T.faint,
        }}>
          <span style={{
            padding: '4px 10px', border: `1px solid ${T.rule}`,
            background: T.panel, color: T.ink,
          }}>scalar</span>
          <span style={{
            padding: '4px 10px', border: `1px solid ${T.rule}`,
          }}>convex</span>
        </div>
      </div>

      {/* Hero */}
      <div style={{
        padding: '20px 56px 18px',
        borderBottom: `1px solid ${T.rule}`,
        display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 48,
        alignItems: 'end',
      }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
            letterSpacing: '0.2em', textTransform: 'uppercase',
            color: T.faint, marginBottom: 8,
          }}>§ 4.1</div>
          <h1 style={{
            margin: 0, fontWeight: 600, letterSpacing: '-0.025em',
            fontSize: 40, lineHeight: 1.02, color: T.ink,
          }}>
            Following the gradient<br/>
            <span style={{ color: T.faint, fontWeight: 400 }}>step by step.</span>
          </h1>
        </div>
        <p style={{
          margin: 0, color: T.faint, fontSize: 15, lineHeight: 1.55,
          paddingBottom: 6, maxWidth: 460,
        }}>
          One parameter, one loss surface, one learning rate. The simplest <strong style={{ color: T.ink, fontWeight: 600 }}>first-order optimizer</strong> reduces to a single update rule — but the choice of <ColorWord c={T.alpha}>α</ColorWord> decides whether the iterates converge, oscillate, or fly to infinity.
        </p>
      </div>

      {/* Definition block */}
      <div style={{
        padding: '14px 56px',
        borderBottom: `1px solid ${T.rule}`,
        background: T.panel,
        display: 'grid', gridTemplateColumns: '160px 1fr 1fr',
        alignItems: 'center', gap: 28,
      }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
            letterSpacing: '0.18em', textTransform: 'uppercase',
            color: T.faint,
          }}>Definition 4.1</div>
          <div style={{
            fontWeight: 600, fontSize: 17, marginTop: 4,
            color: T.ink, letterSpacing: '-0.01em',
          }}>Update rule</div>
        </div>
        <div style={{ fontSize: 24, color: T.ink }}>
          <K tex="\textcolor{#1f4ae2}{w_{t+1}} = \textcolor{#1f4ae2}{w_t} - \textcolor{#c2410c}{\alpha}\, \textcolor{#7c3aed}{\nabla \mathcal{L}(w_t)}" />
        </div>
        <div style={{ fontSize: 14, color: T.faint, lineHeight: 1.5 }}>
          With <ColorWord c={T.loss}>ℒ(w)=w²</ColorWord> the gradient is closed-form:{' '}
          <span style={{ fontSize: 16 }}><K tex="\textcolor{#7c3aed}{\nabla \mathcal{L}} = 2w" /></span>.
        </div>
      </div>

      {/* Body */}
      <div style={{
        flex: 1,
        padding: '20px 56px',
        display: 'grid',
        gridTemplateColumns: '1fr 360px',
        gap: 24,
        minHeight: 0,
      }}>
        {/* Left side — figure + history below */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <TextbookFigure here={here} grad={grad} history={history} stepIdx={stepIdx} />
          <TextbookHistoryRow history={history} stepIdx={stepIdx} />
        </div>

        {/* Right side — metrics + controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, minHeight: 0 }}>
          <TextbookMetrics here={here} grad={grad} stepIdx={stepIdx} />
          <TextbookControls />
        </div>
      </div>
    </div>
  );
}

function ColorWord({ c, children }) {
  return <span style={{ color: c, fontWeight: 600 }}>{children}</span>;
}

function TextbookFigure({ here, grad, history, stepIdx }) {
  const W = 780, H = 300, PX = 56, PY = 24;
  const xs = (w) => PX + ((w + 4.2) / 8.4) * (W - 2 * PX);
  const ys = (L) => H - PY - ((L + 1) / 19) * (H - 2 * PY);
  const curvePts = [];
  for (let w = -4.2; w <= 4.2; w += 0.05) curvePts.push(`${xs(w).toFixed(2)},${ys(w * w).toFixed(2)}`);
  const slope = grad;
  const tx0 = here.w - 1.6, tx1 = here.w + 1.6;
  const ty0 = here.L + slope * (tx0 - here.w);
  const ty1 = here.L + slope * (tx1 - here.w);

  // shaded region from current to next step
  const next = history[stepIdx + 1];

  return (
    <figure style={{
      margin: 0,
      border: `1px solid ${T.rule}`,
      background: T.bg,
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '12px 20px',
        borderBottom: `1px solid ${T.rule}`,
      }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
            letterSpacing: '0.18em', textTransform: 'uppercase', color: T.faint,
          }}>Figure 1</div>
          <div style={{ fontSize: 14, fontWeight: 600, marginTop: 2 }}>
            Descent trajectory on <K tex="\mathcal{L}(w) = w^2" />
          </div>
        </div>
        <Legend />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
        {/* very faint grid */}
        {[-4, -3, -2, -1, 1, 2, 3, 4].map((t) => (
          <line key={`v${t}`} x1={xs(t)} x2={xs(t)} y1={PY} y2={H - PY}
            stroke={T.rule} strokeWidth="1" />
        ))}
        {[4, 8, 12, 16].map((t) => (
          <line key={`h${t}`} x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)}
            stroke={T.rule} strokeWidth="1" />
        ))}

        {/* axes */}
        <line x1={PX} x2={W - PX} y1={ys(0)} y2={ys(0)} stroke={T.ink} strokeWidth="1.2" />
        <line x1={xs(0)} x2={xs(0)} y1={PY} y2={H - PY} stroke={T.ink} strokeWidth="1.2" />

        {[-4, -2, 2, 4].map((t) => (
          <text key={`tx${t}`} x={xs(t)} y={ys(0) + 18}
            fontSize="11" fontFamily="JetBrains Mono, monospace"
            textAnchor="middle" fill={T.faint}>{t}</text>
        ))}
        {[4, 8, 12, 16].map((t) => (
          <text key={`ty${t}`} x={xs(0) - 8} y={ys(t) + 4}
            fontSize="11" fontFamily="JetBrains Mono, monospace"
            textAnchor="end" fill={T.faint}>{t}</text>
        ))}
        <text x={W - PX + 4} y={ys(0) + 16} fontSize="12"
          fontFamily="JetBrains Mono, monospace" fill={T.ink}>w</text>
        <text x={xs(0) + 8} y={PY + 12} fontSize="12"
          fontFamily="JetBrains Mono, monospace" fill={T.ink}>ℒ</text>

        {/* loss curve */}
        <polyline points={curvePts.join(' ')} fill="none"
          stroke={T.loss} strokeWidth="2.2" />

        {/* tangent (gradient direction) */}
        <line x1={xs(tx0)} y1={ys(ty0)} x2={xs(tx1)} y2={ys(ty1)}
          stroke={T.grad} strokeWidth="1.6" strokeDasharray="6 4" />

        {/* trail with connecting chords */}
        {history.slice(0, stepIdx + 1).map((p, i, arr) => {
          const nx = arr[i + 1];
          return (
            <g key={i}>
              {nx && (
                <line x1={xs(p.w)} y1={ys(p.L)} x2={xs(nx.w)} y2={ys(nx.L)}
                  stroke={T.w} strokeWidth="1.2" opacity="0.4" />
              )}
              <circle cx={xs(p.w)} cy={ys(p.L)} r={i === stepIdx ? 0 : 3.5}
                fill={T.bg} stroke={T.w} strokeWidth="1.6" />
            </g>
          );
        })}

        {/* projected next step */}
        {next && (
          <g opacity="0.5">
            <line x1={xs(here.w)} y1={ys(here.L)} x2={xs(next.w)} y2={ys(next.L)}
              stroke={T.alpha} strokeWidth="1.5" strokeDasharray="3 3" />
            <circle cx={xs(next.w)} cy={ys(next.L)} r="3.5"
              fill={T.bg} stroke={T.alpha} strokeWidth="1.4" />
          </g>
        )}

        {/* current step — bullseye */}
        <circle cx={xs(here.w)} cy={ys(here.L)} r="10" fill={T.bg}
          stroke={T.w} strokeWidth="1.5" />
        <circle cx={xs(here.w)} cy={ys(here.L)} r="5" fill={T.w} />

        {/* dropline + axis label */}
        <line x1={xs(here.w)} y1={ys(here.L)} x2={xs(here.w)} y2={ys(0)}
          stroke={T.w} strokeWidth="1" strokeDasharray="2 3" opacity="0.5" />
        <rect x={xs(here.w) - 18} y={ys(0) + 22} width={36} height={18}
          fill={T.w} />
        <text x={xs(here.w)} y={ys(0) + 35}
          fontSize="11" fontFamily="JetBrains Mono, monospace"
          textAnchor="middle" fill={T.bg} fontWeight="600">
          {here.w.toFixed(2)}
        </text>

        {/* gradient label */}
        <g>
          <rect x={xs(tx1) - 4} y={ys(ty1) - 24} width={104} height={22}
            fill={T.bg} stroke={T.grad} strokeWidth="1" />
          <text x={xs(tx1) + 4} y={ys(ty1) - 9}
            fontSize="11" fontFamily="JetBrains Mono, monospace" fill={T.grad}>
            slope = {grad.toFixed(2)}
          </text>
        </g>
      </svg>

      <figcaption style={{
        padding: '12px 20px',
        borderTop: `1px solid ${T.rule}`,
        fontSize: 13, lineHeight: 1.55, color: T.faint,
      }}>
        At step {stepIdx}, the parameter <ColorWord c={T.w}>w = {here.w.toFixed(3)}</ColorWord> sits on the curve at <ColorWord c={T.loss}>ℒ = {here.L.toFixed(3)}</ColorWord>. The tangent shows <ColorWord c={T.grad}>∇ℒ = {grad.toFixed(3)}</ColorWord>; multiplied by <ColorWord c={T.alpha}>α = 0.25</ColorWord> and subtracted, it produces the next iterate.
      </figcaption>
    </figure>
  );
}

function Legend() {
  return (
    <div style={{
      display: 'flex', gap: 18,
      fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
      letterSpacing: '0.06em', color: T.faint,
    }}>
      <LegendItem dot={T.loss} solid label="ℒ(w)" />
      <LegendItem dot={T.grad} dashed label="∇ℒ" />
      <LegendItem dot={T.w} solid label="trajectory" />
      <LegendItem dot={T.alpha} dashed label="next step" />
    </div>
  );
}

function LegendItem({ dot, dashed, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <svg width="22" height="6">
        <line x1="0" y1="3" x2="22" y2="3" stroke={dot} strokeWidth="2"
          strokeDasharray={dashed ? '4 2' : 'none'} />
      </svg>
      <span style={{ color: T.ink }}>{label}</span>
    </div>
  );
}

function TextbookMetrics({ here, grad, stepIdx }) {
  const K = window.K;
  const cells = [
    { label: 'iteration', tex: 't', val: String(stepIdx), color: T.ink },
    { label: 'weight', tex: 'w_t', val: here.w.toFixed(3), color: T.w },
    { label: 'gradient', tex: '\\nabla\\mathcal{L}', val: grad.toFixed(3), color: T.grad },
    { label: 'loss', tex: '\\mathcal{L}(w_t)', val: here.L.toFixed(3), color: T.loss },
  ];
  return (
    <div style={{
      border: `1px solid ${T.rule}`,
      display: 'grid', gridTemplateColumns: '1fr 1fr',
    }}>
      {cells.map((c, i) => (
        <div key={c.label} style={{
          padding: '16px 18px',
          borderRight: i % 2 === 0 ? `1px solid ${T.rule}` : 'none',
          borderBottom: i < 2 ? `1px solid ${T.rule}` : 'none',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: T.faint, marginBottom: 6,
          }}>
            <span style={{
              width: 6, height: 6, background: c.color, borderRadius: '50%',
            }} />
            {c.label}
          </div>
          <div style={{ fontSize: 13, color: T.faint, marginBottom: 2 }}>
            <K tex={c.tex} />
          </div>
          <div style={{
            fontSize: 28, fontWeight: 600, color: c.color,
            letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums',
          }}>{c.val}</div>
        </div>
      ))}
    </div>
  );
}

function TextbookControls() {
  const K = window.K;
  return (
    <div style={{
      flex: 1,
      border: `1px solid ${T.rule}`,
      display: 'flex', flexDirection: 'column',
      minHeight: 0,
    }}>
      <div style={{
        padding: '12px 18px',
        borderBottom: `1px solid ${T.rule}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
          letterSpacing: '0.18em', textTransform: 'uppercase', color: T.faint,
        }}>Hyperparameters</div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          color: T.loss, fontWeight: 600,
        }}>● converging</div>
      </div>

      <div style={{ padding: '18px 18px 8px', flex: 1 }}>
        <TbSlider label="Learning rate" tex="\alpha" value="0.25" pos={0.25}
          accent={T.alpha} status={{ tone: 'ok', text: 'within stable region' }} />
        <TbSlider label="Initial weight" tex="w_0" value="+3.00" pos={0.8}
          accent={T.w} status={{ tone: 'note', text: 'far from minimum' }} />

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6,
          marginTop: 14,
        }}>
          {[
            { label: 'slow', val: 0.01 },
            { label: 'good', val: 0.25, active: true },
            { label: 'osc.', val: 0.95 },
          ].map((p) => (
            <button key={p.label} style={{
              padding: '8px 6px',
              border: `1px solid ${p.active ? T.ink : T.rule}`,
              background: p.active ? T.ink : T.bg,
              color: p.active ? T.bg : T.ink,
              fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
              cursor: 'pointer', textAlign: 'left',
              display: 'flex', flexDirection: 'column', gap: 2,
            }}>
              <span style={{ opacity: 0.7, fontSize: 9, letterSpacing: '0.12em', textTransform: 'uppercase' }}>{p.label}</span>
              <span style={{ fontSize: 13, fontWeight: 600 }}>α = {p.val}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: '1fr auto',
        borderTop: `1px solid ${T.rule}`,
      }}>
        <button style={{
          background: T.ink, color: T.bg, border: 'none',
          padding: '14px 18px',
          fontFamily: 'Inter, sans-serif', fontSize: 14, fontWeight: 600,
          letterSpacing: '-0.01em', cursor: 'pointer', textAlign: 'left',
        }}>Run gradient descent →</button>
        <button style={{
          background: T.bg, color: T.ink,
          border: 'none', borderLeft: `1px solid ${T.rule}`,
          padding: '14px 18px',
          fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
          letterSpacing: '0.1em', textTransform: 'uppercase', cursor: 'pointer',
        }}>Reset</button>
      </div>
    </div>
  );
}

function TbSlider({ label, tex, value, pos, accent, status }) {
  const K = window.K;
  const dotColor = status.tone === 'ok' ? T.loss : status.tone === 'warn' ? T.alpha : T.faint;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        marginBottom: 8,
      }}>
        <div style={{ fontSize: 13, color: T.ink }}>
          {label} <span style={{ color: T.faint, marginLeft: 4 }}><K tex={tex} /></span>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 14, color: accent, fontWeight: 600,
          fontVariantNumeric: 'tabular-nums',
        }}>{value}</div>
      </div>
      <div style={{
        position: 'relative', height: 4,
        background: T.rule,
      }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${pos * 100}%`, background: accent,
        }} />
        <div style={{
          position: 'absolute', left: `${pos * 100}%`, top: -5, bottom: -5,
          width: 14, background: T.bg, border: `2px solid ${accent}`,
          transform: 'translateX(-50%)',
        }} />
      </div>
      <div style={{
        marginTop: 8, display: 'flex', alignItems: 'center', gap: 6,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
        letterSpacing: '0.08em', color: T.faint,
      }}>
        <span style={{ width: 5, height: 5, background: dotColor, borderRadius: '50%' }} />
        {status.text}
      </div>
    </div>
  );
}

function TextbookHistoryRow({ history, stepIdx }) {
  return (
    <div style={{
      border: `1px solid ${T.rule}`,
      display: 'grid', gridTemplateColumns: '1.4fr 1fr',
    }}>
      <div style={{ borderRight: `1px solid ${T.rule}` }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '10px 18px',
          borderBottom: `1px solid ${T.rule}`,
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
            letterSpacing: '0.18em', textTransform: 'uppercase', color: T.faint,
          }}>Figure 2 — Loss history</div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
            color: T.faint, fontVariantNumeric: 'tabular-nums',
          }}>
            ℒ₀ = {history[0].L.toFixed(2)} &nbsp;→&nbsp; ℒ₁₂ = {history[history.length - 1].L.toFixed(4)}
          </div>
        </div>
        <TbHistorySvg history={history} stepIdx={stepIdx} />
      </div>
      <TbStepTable history={history} stepIdx={stepIdx} />
    </div>
  );
}

function TbHistorySvg({ history, stepIdx }) {
  const W = 460, H = 130, PX = 36, PY = 16;
  const maxL = Math.max(...history.map((p) => p.L)) || 1;
  const xs = (i) => PX + (i / (history.length - 1)) * (W - 2 * PX);
  const ys = (L) => H - PY - (L / maxL) * (H - 2 * PY);
  const linePts = history.map((p) => `${xs(p.i).toFixed(2)},${ys(p.L).toFixed(2)}`).join(' ');
  const areaPts = `${xs(0)},${ys(0)} ${linePts} ${xs(history.length - 1)},${ys(0)}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      {[0, 3, 6, 9].map((t) => (
        <g key={t}>
          <line x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)} stroke={T.rule} />
          <text x={PX - 6} y={ys(t) + 4} fontSize="10" textAnchor="end"
            fontFamily="JetBrains Mono, monospace" fill={T.faint}>{t}</text>
        </g>
      ))}
      <polygon points={areaPts} fill={T.loss} opacity="0.08" />
      <polyline points={linePts} fill="none" stroke={T.loss} strokeWidth="2" />
      {history.map((p, i) => (
        <circle key={i} cx={xs(p.i)} cy={ys(p.L)}
          r={i === stepIdx ? 5 : 2.4}
          fill={i === stepIdx ? T.bg : T.loss}
          stroke={i === stepIdx ? T.loss : 'none'} strokeWidth="2" />
      ))}
      {history.map((p, i) => i % 3 === 0 && (
        <text key={`x${i}`} x={xs(p.i)} y={H - 4}
          fontSize="10" textAnchor="middle"
          fontFamily="JetBrains Mono, monospace" fill={T.faint}>{i}</text>
      ))}
    </svg>
  );
}

function TbStepTable({ history, stepIdx }) {
  const rows = history.slice(0, 6);
  return (
    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
      <div style={{
        display: 'grid', gridTemplateColumns: '36px 1fr 1fr',
        padding: '10px 14px',
        color: T.faint, fontSize: 9, letterSpacing: '0.14em', textTransform: 'uppercase',
        borderBottom: `1px solid ${T.rule}`,
      }}>
        <span>t</span><span>w_t</span><span style={{ textAlign: 'right' }}>ℒ(w_t)</span>
      </div>
      {rows.map((p, i) => (
        <div key={i} style={{
          display: 'grid', gridTemplateColumns: '36px 1fr 1fr',
          padding: '5px 14px',
          background: i === stepIdx ? `${T.w}0e` : 'transparent',
          color: i === stepIdx ? T.ink : T.faint,
          fontWeight: i === stepIdx ? 600 : 400,
        }}>
          <span>{String(p.i).padStart(2, '0')}</span>
          <span style={{ color: i === stepIdx ? T.w : T.ink, fontWeight: i === stepIdx ? 600 : 500 }}>
            {(p.w >= 0 ? '+' : '') + p.w.toFixed(3)}
          </span>
          <span style={{
            textAlign: 'right',
            color: i === stepIdx ? T.loss : T.ink,
            fontWeight: i === stepIdx ? 600 : 500,
          }}>{p.L.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

window.TextbookVariant = TextbookVariant;
