// Direction B — Engineering notebook: cream paper with dotted grid, monospace
// labels, a single oxblood accent. Reads like a scanned engineer's worksheet.

const N = {
  paper: '#f3ede0',
  panel: '#f8f3e6',
  ink: '#171513',
  faint: '#5b5447',
  rule: '#c8bda3',
  hair: '#9d927b',
  accent: '#8a1d1d',   // oxblood / engineer's red pen
  blueprint: '#1d3a6e',
};

function NotebookVariant() {
  window.useKatexReady();
  const K = window.K;
  const history = window.simulate(3.0, 0.25, 12);
  const stepIdx = 4;
  const here = history[stepIdx];
  const grad = 2 * here.w;

  return (
    <div style={{
      width: 1280, height: 920,
      background: N.paper,
      color: N.ink,
      fontFamily: 'Inter, system-ui, sans-serif',
      overflow: 'hidden',
      position: 'relative',
      backgroundImage:
        `radial-gradient(${N.hair}33 1px, transparent 1.2px)`,
      backgroundSize: '22px 22px',
      backgroundPosition: '0 0',
    }}>
      {/* Margin rule */}
      <div style={{
        position: 'absolute', left: 88, top: 0, bottom: 0,
        borderLeft: `1px solid ${N.accent}55`,
      }} />
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 88,
        background: `linear-gradient(to right, ${N.paper}, ${N.paper}cc)`,
      }} />

      {/* Header strip */}
      <div style={{
        padding: '28px 56px 18px 112px',
        borderBottom: `1px solid ${N.rule}`,
        display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
        background: `${N.paper}ee`,
      }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 11, letterSpacing: '0.18em', textTransform: 'uppercase',
            color: N.faint, marginBottom: 4,
          }}>
            NB-04 / OPTIM &nbsp;·&nbsp; 2026-05-19 &nbsp;·&nbsp; PG. 14
          </div>
          <h1 style={{
            fontFamily: 'Inter, sans-serif',
            fontWeight: 700, fontSize: 38, lineHeight: 1,
            letterSpacing: '-0.02em',
            margin: 0, color: N.ink,
          }}>
            Gradient descent — scalar case
          </h1>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 12, color: N.faint, textAlign: 'right',
          lineHeight: 1.5,
        }}>
          objective &nbsp; <span style={{ color: N.ink }}>min ℒ(w)</span><br/>
          method &nbsp;&nbsp;&nbsp; <span style={{ color: N.ink }}>1st-order</span><br/>
          stamp &nbsp;&nbsp;&nbsp;&nbsp; <span style={{ color: N.accent }}>● running</span>
        </div>
      </div>

      {/* Body */}
      <div style={{
        position: 'relative',
        padding: '28px 56px 28px 112px',
        height: 'calc(100% - 110px)',
        display: 'grid',
        gridTemplateColumns: '1.55fr 1fr',
        gridTemplateRows: 'auto 1fr',
        gap: 22,
      }}>
        {/* Update rule strip — spans top */}
        <div style={{
          gridColumn: '1 / -1',
          border: `1px solid ${N.hair}`,
          background: N.panel,
          padding: '14px 22px',
          display: 'flex', alignItems: 'center', gap: 28,
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 10, letterSpacing: '0.16em', textTransform: 'uppercase',
            color: N.accent, fontWeight: 600,
            borderRight: `1px solid ${N.rule}`, paddingRight: 22,
          }}>(eq. 1)</div>
          <div style={{ fontSize: 22, flex: 1 }}>
            <K tex="w_{t+1} \;=\; w_t \;-\; \alpha \cdot \nabla\mathcal{L}(w_t) \qquad \mathcal{L}(w) = w^{2}, \;\; \nabla\mathcal{L} = 2w" />
          </div>
        </div>

        {/* Left — plot */}
        <NotebookLandscape here={here} grad={grad} history={history} stepIdx={stepIdx} />

        {/* Right — control + log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <NotebookControls />
          <NotebookLog history={history} stepIdx={stepIdx} />
        </div>
      </div>
    </div>
  );
}

function NotebookLandscape({ here, grad, history, stepIdx }) {
  const K = window.K;
  const W = 700, H = 460, PX = 64, PY = 32;
  const xs = (w) => PX + ((w + 4.2) / 8.4) * (W - 2 * PX);
  const ys = (L) => H - PY - ((L + 1) / 19) * (H - 2 * PY);
  const curvePts = [];
  for (let w = -4.2; w <= 4.2; w += 0.05) curvePts.push(`${xs(w).toFixed(2)},${ys(w * w).toFixed(2)}`);

  const slope = grad;
  const tx0 = here.w - 1.6, tx1 = here.w + 1.6;
  const ty0 = here.L + slope * (tx0 - here.w);
  const ty1 = here.L + slope * (tx1 - here.w);

  const trail = history.slice(0, stepIdx);

  return (
    <div style={{
      border: `1px solid ${N.hair}`, background: N.panel,
      padding: 0, position: 'relative',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* tab header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 18px',
        borderBottom: `1px solid ${N.rule}`,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
        color: N.faint, letterSpacing: '0.1em', textTransform: 'uppercase',
      }}>
        <span>fig. 1 &nbsp;·&nbsp; loss landscape ℒ(w) = w²</span>
        <span>t = {stepIdx} / {history.length - 1}</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', flex: 1 }}>
        {/* fine engineering grid */}
        <defs>
          <pattern id="nbgrid" x={PX} y={PY} width="22" height="22" patternUnits="userSpaceOnUse">
            <path d="M 22 0 L 0 0 0 22" fill="none" stroke={`${N.hair}55`} strokeWidth="0.5" />
          </pattern>
          <pattern id="nbgrid5" x={PX} y={PY} width="110" height="110" patternUnits="userSpaceOnUse">
            <path d="M 110 0 L 0 0 0 110" fill="none" stroke={`${N.hair}aa`} strokeWidth="0.7" />
          </pattern>
        </defs>
        <rect x={PX} y={PY} width={W - 2*PX} height={H - 2*PY} fill="url(#nbgrid)" />
        <rect x={PX} y={PY} width={W - 2*PX} height={H - 2*PY} fill="url(#nbgrid5)" />

        {/* axis ticks */}
        {[-4, -3, -2, -1, 0, 1, 2, 3, 4].map((t) => (
          <g key={`tk${t}`}>
            <line x1={xs(t)} x2={xs(t)} y1={ys(0) - 4} y2={ys(0) + 4} stroke={N.ink} strokeWidth="1" />
            <text x={xs(t)} y={ys(0) + 18} fontSize="11"
              fontFamily="JetBrains Mono, monospace" textAnchor="middle" fill={N.faint}>{t}</text>
          </g>
        ))}
        {[2, 4, 6, 8, 10, 12, 14, 16].map((t) => (
          <g key={`lk${t}`}>
            <line x1={xs(0) - 4} x2={xs(0) + 4} y1={ys(t)} y2={ys(t)} stroke={N.ink} strokeWidth="1" />
            <text x={xs(0) - 8} y={ys(t) + 4} fontSize="11"
              fontFamily="JetBrains Mono, monospace" textAnchor="end" fill={N.faint}>{t}</text>
          </g>
        ))}

        {/* axes */}
        <line x1={PX} x2={W - PX} y1={ys(0)} y2={ys(0)} stroke={N.ink} strokeWidth="1.4" />
        <line x1={xs(0)} x2={xs(0)} y1={PY} y2={H - PY} stroke={N.ink} strokeWidth="1.4" />
        <polygon points={`${W - PX},${ys(0)} ${W - PX - 8},${ys(0) - 4} ${W - PX - 8},${ys(0) + 4}`} fill={N.ink} />
        <polygon points={`${xs(0)},${PY} ${xs(0) - 4},${PY + 8} ${xs(0) + 4},${PY + 8}`} fill={N.ink} />

        <text x={W - PX + 8} y={ys(0) + 4} fontSize="13"
          fontFamily="JetBrains Mono, monospace" fill={N.ink}>w</text>
        <text x={xs(0) + 8} y={PY - 4} fontSize="13"
          fontFamily="JetBrains Mono, monospace" fill={N.ink}>ℒ</text>

        {/* loss curve — drawn like a pen line, two passes for slight texture */}
        <polyline points={curvePts.join(' ')} fill="none" stroke={N.blueprint} strokeWidth="1.6" />
        <polyline points={curvePts.join(' ')} fill="none" stroke={N.blueprint} strokeWidth="0.6" opacity="0.45"
          transform="translate(0.5,0.5)" />

        {/* tangent (red pen annotation) */}
        <line x1={xs(tx0)} y1={ys(ty0)} x2={xs(tx1)} y2={ys(ty1)}
          stroke={N.accent} strokeWidth="1.4" strokeDasharray="5 3" />

        {/* trail */}
        {trail.map((p, i) => (
          <g key={i}>
            <line x1={xs(p.w)} y1={ys(p.L)}
              x2={xs(trail[i + 1] ? trail[i + 1].w : here.w)}
              y2={ys(trail[i + 1] ? trail[i + 1].L : here.L)}
              stroke={N.accent} strokeWidth="0.8" strokeDasharray="2 3" opacity="0.55" />
            <circle cx={xs(p.w)} cy={ys(p.L)} r="2.5" fill={N.paper} stroke={N.faint} strokeWidth="1" />
          </g>
        ))}

        {/* current point — hand-marked X */}
        <g>
          <line x1={xs(here.w) - 7} y1={ys(here.L) - 7}
            x2={xs(here.w) + 7} y2={ys(here.L) + 7}
            stroke={N.accent} strokeWidth="1.6" />
          <line x1={xs(here.w) - 7} y1={ys(here.L) + 7}
            x2={xs(here.w) + 7} y2={ys(here.L) - 7}
            stroke={N.accent} strokeWidth="1.6" />
          <circle cx={xs(here.w)} cy={ys(here.L)} r="10" fill="none"
            stroke={N.accent} strokeWidth="1" opacity="0.5" />
        </g>

        {/* gradient annotation with arrow */}
        <g>
          <path
            d={`M ${xs(here.w) + 18} ${ys(here.L) - 24}
                Q ${xs(here.w) + 90} ${ys(here.L) - 80}
                  ${xs(here.w) + 150} ${ys(here.L) - 70}`}
            fill="none" stroke={N.accent} strokeWidth="0.9" />
          <text x={xs(here.w) + 154} y={ys(here.L) - 66}
            fontSize="11" fontFamily="JetBrains Mono, monospace" fill={N.accent}>
            ∇ℒ = 2·{here.w.toFixed(2)} = {grad.toFixed(2)}
          </text>
        </g>

        {/* dropline to axis */}
        <line x1={xs(here.w)} y1={ys(here.L)} x2={xs(here.w)} y2={ys(0)}
          stroke={N.faint} strokeWidth="0.6" strokeDasharray="2 3" />
        <text x={xs(here.w) + 4} y={ys(0) - 6}
          fontSize="10" fontFamily="JetBrains Mono, monospace" fill={N.faint}>
          w₄
        </text>
      </svg>

      {/* footer caption */}
      <div style={{
        borderTop: `1px solid ${N.rule}`,
        padding: '10px 18px',
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 16,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
        color: N.faint,
      }}>
        <NbStat label="t" val={String(stepIdx)} />
        <NbStat label="w" val={here.w.toFixed(3)} />
        <NbStat label="∇ℒ" val={grad.toFixed(3)} />
        <NbStat label="ℒ" val={here.L.toFixed(3)} mark />
      </div>
    </div>
  );
}

function NbStat({ label, val, mark }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
      <span style={{ color: N.faint }}>{label}</span>
      <span style={{
        flex: 1, borderBottom: `1px dotted ${N.hair}`, marginBottom: 4,
      }} />
      <span style={{
        color: mark ? N.accent : N.ink, fontWeight: 600,
      }}>{val}</span>
    </div>
  );
}

function NotebookControls() {
  const K = window.K;
  return (
    <div style={{
      border: `1px solid ${N.hair}`, background: N.panel,
    }}>
      <div style={{
        padding: '10px 18px',
        borderBottom: `1px solid ${N.rule}`,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
        color: N.faint, letterSpacing: '0.1em', textTransform: 'uppercase',
      }}>parameters</div>
      <div style={{ padding: '18px 22px' }}>
        <NbDial label="learning rate" tex="\alpha" value="0.25" pos={0.25} flag="ok" />
        <NbDial label="initial weight" tex="w_0" value="+3.00" pos={0.8} flag="ok" />

        <div style={{
          marginTop: 6, padding: '10px 12px',
          borderLeft: `2px solid ${N.accent}`,
          background: `${N.accent}0d`,
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          color: N.faint, lineHeight: 1.6,
        }}>
          <div style={{ color: N.accent, fontWeight: 600, marginBottom: 4 }}>NOTE</div>
          α ≥ 1.00 → diverge<br/>
          α = 0.95 → oscillate<br/>
          α ≤ 0.05 → slow
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          <button style={{
            flex: 1, background: N.ink, color: N.paper,
            border: 'none', padding: '10px 14px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
            cursor: 'pointer',
          }}>▷ run</button>
          <button style={{
            background: 'transparent', color: N.ink,
            border: `1px solid ${N.ink}`, padding: '10px 14px',
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
            cursor: 'pointer',
          }}>↺ reset</button>
        </div>
      </div>
    </div>
  );
}

function NbDial({ label, tex, value, pos }) {
  const K = window.K;
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
        marginBottom: 6,
      }}>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
          letterSpacing: '0.08em', color: N.faint,
        }}>
          {label} &nbsp;<span style={{ color: N.ink }}><K tex={tex} /></span>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
          color: N.ink, fontWeight: 600,
        }}>{value}</div>
      </div>
      <div style={{
        position: 'relative', height: 22,
        borderTop: `1px solid ${N.ink}`,
        borderBottom: `1px solid ${N.ink}`,
        background: `repeating-linear-gradient(to right, transparent 0 9px, ${N.hair}66 9px 10px)`,
      }}>
        <div style={{
          position: 'absolute', left: `${pos * 100}%`, top: -3, bottom: -3,
          width: 2, background: N.accent,
          transform: 'translateX(-50%)',
        }} />
        <div style={{
          position: 'absolute', left: `${pos * 100}%`, top: -7,
          fontFamily: 'JetBrains Mono, monospace', fontSize: 9,
          color: N.accent, transform: 'translateX(-50%)',
          background: N.panel, padding: '0 3px',
        }}>▼</div>
      </div>
    </div>
  );
}

function NotebookLog({ history, stepIdx }) {
  return (
    <div style={{
      flex: 1,
      border: `1px solid ${N.hair}`, background: N.panel,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      <div style={{
        padding: '10px 18px',
        borderBottom: `1px solid ${N.rule}`,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
        color: N.faint, letterSpacing: '0.1em', textTransform: 'uppercase',
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span>iteration log</span>
        <span>fig. 2</span>
      </div>

      {/* Mini history sparkline */}
      <div style={{ padding: '8px 18px', borderBottom: `1px solid ${N.rule}` }}>
        <NbSparkline history={history} stepIdx={stepIdx} />
      </div>

      <div style={{
        flex: 1, overflow: 'hidden',
        fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
        padding: '8px 0',
      }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '40px 1fr 1fr 1fr',
          padding: '6px 18px',
          color: N.faint, fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
          borderBottom: `1px solid ${N.rule}`,
        }}>
          <span>t</span><span>w_t</span><span>∇ℒ</span><span style={{ textAlign: 'right' }}>ℒ(w)</span>
        </div>
        {history.slice(0, 7).map((p, i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '40px 1fr 1fr 1fr',
            padding: '4px 18px',
            color: i === stepIdx ? N.accent : N.ink,
            background: i === stepIdx ? `${N.accent}11` : 'transparent',
            fontVariantNumeric: 'tabular-nums',
          }}>
            <span style={{ color: N.faint }}>{String(p.i).padStart(2, '0')}</span>
            <span>{(p.w >= 0 ? '+' : '') + p.w.toFixed(3)}</span>
            <span>{((2 * p.w) >= 0 ? '+' : '') + (2 * p.w).toFixed(3)}</span>
            <span style={{ textAlign: 'right' }}>{p.L.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function NbSparkline({ history, stepIdx }) {
  const W = 360, H = 56;
  const maxL = Math.max(...history.map((p) => p.L)) || 1;
  const xs = (i) => (i / (history.length - 1)) * W;
  const ys = (L) => H - (L / maxL) * (H - 8) - 4;
  const pts = history.map((p) => `${xs(p.i).toFixed(2)},${ys(p.L).toFixed(2)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <polyline points={pts} fill="none" stroke={N.blueprint} strokeWidth="1.5" />
      {history.map((p, i) => (
        <circle key={i} cx={xs(p.i)} cy={ys(p.L)} r={i === stepIdx ? 3 : 1.6}
          fill={i === stepIdx ? N.accent : N.blueprint} />
      ))}
    </svg>
  );
}

window.NotebookVariant = NotebookVariant;
