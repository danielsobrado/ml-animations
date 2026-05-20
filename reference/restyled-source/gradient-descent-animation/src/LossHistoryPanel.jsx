import React from 'react';
import { Figure, Eq } from './_design-system/ui';

export default function LossHistoryPanel({ history = [], current }) {
    const W = 460, H = 200, PX = 40, PY = 22;

    if (history.length === 0) {
        return (
            <Figure
                label="Figure 2"
                title="Loss history"
                caption={<><span className="lead">Figure 2.</span>{' '}
                    Runs ▶ to plot <Eq tex="\mathcal{L}(w_t)" /> across iterations <Eq tex="t" />.</>}
            >
                <div className="gd-history-empty">No data yet.</div>
            </Figure>
        );
    }

    const maxL = Math.max(...history.map((h) => h.loss), 1);
    const maxIter = Math.max(history.length - 1, 1);
    const xs = (i) => PX + (i / maxIter) * (W - 2 * PX);
    const ys = (L) => H - PY - (L / maxL) * (H - 2 * PY);

    const linePts = history.map((p) => `${xs(p.iteration).toFixed(2)},${ys(p.loss).toFixed(2)}`).join(' ');

    // Choose ~4 evenly-spaced y-ticks
    const yTickVals = [0, maxL * 0.33, maxL * 0.66, maxL].map((v) => +v.toFixed(2));
    const xTickStep = Math.max(1, Math.ceil(maxIter / 6));

    const finalLoss = history[history.length - 1]?.loss ?? 0;

    return (
        <Figure
            label="Figure 2"
            title="Loss history"
            right={<span>{history.length} iterations</span>}
            caption={
                <>
                    <span className="lead">Figure 2.</span>{' '}
                    <Eq tex="\mathcal{L}" /> decays geometrically when <Eq tex="\alpha < 1" />; the run ended at{' '}
                    <Eq tex={`\\mathcal{L} = ${finalLoss.toFixed(4)}`} />.
                </>
            }
        >
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
                {/* Y gridlines */}
                {yTickVals.map((t) => (
                    <g key={`yt${t}`}>
                        <line x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)} stroke="var(--ds-grid)" />
                        <text x={PX - 8} y={ys(t) + 4} fontSize="11" textAnchor="end"
                            fontFamily="var(--ds-font-mono)" fill="var(--ds-faint)">{t}</text>
                    </g>
                ))}

                {/* Axes */}
                <line x1={PX} x2={W - PX} y1={H - PY} y2={H - PY}
                    stroke="var(--ds-ink)" strokeWidth="1" />
                <line x1={PX} x2={PX} y1={PY} y2={H - PY}
                    stroke="var(--ds-ink)" strokeWidth="1" />

                {/* X ticks */}
                {Array.from({ length: Math.floor(maxIter / xTickStep) + 1 }, (_, k) => k * xTickStep).map((i) => (
                    <text key={`xt${i}`} x={xs(i)} y={H - PY + 16}
                        fontSize="11" textAnchor="middle"
                        fontFamily="var(--ds-font-mono)" fill="var(--ds-faint)">{i}</text>
                ))}

                {/* Series */}
                <polyline points={linePts} fill="none"
                    stroke="var(--ds-accent)" strokeWidth="1.6" />
                {history.map((p) => (
                    <circle key={p.iteration}
                        cx={xs(p.iteration)} cy={ys(p.loss)}
                        r={p.iteration === current.iteration ? 4.5 : 2.5}
                        fill={p.iteration === current.iteration ? 'var(--ds-warm)' : 'var(--ds-accent)'}
                        stroke={p.iteration === current.iteration ? 'var(--ds-paper)' : 'none'}
                        strokeWidth="1.5"
                    />
                ))}

                {/* Axis labels */}
                <text x={W - PX} y={H - 4} fontSize="11" textAnchor="end"
                    fontStyle="italic" fontFamily="var(--ds-font-serif)" fill="var(--ds-faint)">iteration t</text>
                <text x={PX + 4} y={PY + 4} fontSize="11"
                    fontStyle="italic" fontFamily="var(--ds-font-serif)" fill="var(--ds-faint)"><tspan>ℒ</tspan></text>
            </svg>
        </Figure>
    );
}
