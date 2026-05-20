import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { Figure, Eq } from './_design-system/ui';

// Loss function: ℒ(w) = w²
// Gradient: dℒ/dw = 2w
// Update: w_{t+1} = w_t - α · 2w_t

export default function GradientDescentPanel({
    learningRate,
    startWeight,
    history,
    current,
    onStepChange,
}) {
    const containerRef = useRef(null);
    const ballRef = useRef(null);
    const [width, setWidth] = useState(760);
    const [isRunning, setIsRunning] = useState(false);

    // We animate against a tween-state. React doesn't render every GSAP frame —
    // we drive the SVG element directly via ref while running, then snap to React
    // state when steps complete.
    const tweenStateRef = useRef({ w: startWeight });

    // Track size for responsive width
    useEffect(() => {
        if (!containerRef.current) return;
        const ro = new ResizeObserver((entries) => {
            const cr = entries[0].contentRect;
            if (cr.width) setWidth(cr.width);
        });
        ro.observe(containerRef.current);
        return () => ro.disconnect();
    }, []);

    // Reset the tween-state when params change
    useEffect(() => {
        tweenStateRef.current.w = startWeight;
    }, [startWeight]);

    // --- SVG math --------------------------------------------------------
    const H = 320, PX = 56, PY = 28;
    const W = width;
    const W_RANGE = 4.2;
    const L_RANGE_MAX = 18, L_RANGE_MIN = -1;
    const xs = (w) => PX + ((w + W_RANGE) / (2 * W_RANGE)) * (W - 2 * PX);
    const ys = (L) => H - PY - ((L - L_RANGE_MIN) / (L_RANGE_MAX - L_RANGE_MIN)) * (H - 2 * PY);

    // Curve points (computed once per width)
    const curvePts = React.useMemo(() => {
        const pts = [];
        for (let w = -W_RANGE; w <= W_RANGE; w += 0.04) {
            pts.push(`${xs(w).toFixed(2)},${ys(w * w).toFixed(2)}`);
        }
        return pts.join(' ');
    }, [W]);

    // Ticks
    const xTicks = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
    const yTicks = [0, 4, 8, 12, 16];

    // Current iterate values from React state (used for static markers)
    const w = current.weight;
    const grad = 2 * w;
    const lossHere = w * w;
    const tx0 = w - 1.4, tx1 = w + 1.4;
    const ty0 = lossHere + grad * (tx0 - w);
    const ty1 = lossHere + grad * (tx1 - w);

    // --- Gradient descent loop ------------------------------------------
    const runDescent = async () => {
        if (isRunning) return;
        setIsRunning(true);

        let wCurr = startWeight;
        tweenStateRef.current.w = wCurr;
        onStepChange(0, wCurr);
        // Snap ball to start
        if (ballRef.current) {
            ballRef.current.setAttribute('cx', xs(wCurr));
            ballRef.current.setAttribute('cy', ys(wCurr * wCurr));
        }

        const maxIter = 50;
        const eps = 0.01;

        for (let i = 0; i < maxIter; i++) {
            const g = 2 * wCurr;
            const wNext = wCurr - learningRate * g;

            // Tween the ball via GSAP from current to next position
            await new Promise((resolve) => {
                gsap.to(tweenStateRef.current, {
                    w: wNext,
                    duration: 0.55,
                    ease: 'power2.inOut',
                    onUpdate: () => {
                        const tw = tweenStateRef.current.w;
                        if (ballRef.current) {
                            ballRef.current.setAttribute('cx', xs(tw));
                            ballRef.current.setAttribute('cy', ys(tw * tw));
                        }
                    },
                    onComplete: resolve,
                });
            });

            wCurr = wNext;
            onStepChange(i + 1, wCurr);

            if (Math.abs(wCurr) < eps) break;
            if (Math.abs(wCurr) > 10) break;
            await new Promise((r) => setTimeout(r, 180));
        }

        setIsRunning(false);
    };

    const reset = () => {
        if (isRunning) return;
        tweenStateRef.current.w = startWeight;
        onStepChange(0, startWeight);
        if (ballRef.current) {
            ballRef.current.setAttribute('cx', xs(startWeight));
            ballRef.current.setAttribute('cy', ys(startWeight * startWeight));
        }
    };

    // Trail markers from history (excluding current)
    const trail = history.slice(0, Math.max(0, current.iteration));

    return (
        <Figure
            label={`Figure 1`}
            title="Loss landscape ℒ(w) = w²"
            right={<span>step {current.iteration}</span>}
            caption={
                <>
                    <span className="lead">Figure 1.</span>{' '}
                    The loss surface <Eq tex="\mathcal{L}(w)=w^2" />. The marker shows{' '}
                    <Eq tex={`w_{${current.iteration}} = ${w.toFixed(3)}`} />; the tangent indicates the gradient{' '}
                    <Eq tex="\nabla\mathcal{L}=2w" /> at that point.
                </>
            }
        >
            <div ref={containerRef} className="gd-landscape-wrap">
                <svg
                    viewBox={`0 0 ${W} ${H}`}
                    width="100%" height={H}
                    style={{ display: 'block' }}
                >
                    {/* Faint grid */}
                    {xTicks.map((t) => (
                        <line key={`vx${t}`} x1={xs(t)} x2={xs(t)} y1={PY} y2={H - PY}
                            stroke="var(--ds-grid)" strokeWidth="1" />
                    ))}
                    {yTicks.map((t) => (
                        <line key={`hl${t}`} x1={PX} x2={W - PX} y1={ys(t)} y2={ys(t)}
                            stroke="var(--ds-grid)" strokeWidth="1" />
                    ))}

                    {/* Axes */}
                    <line x1={PX} x2={W - PX} y1={ys(0)} y2={ys(0)}
                        stroke="var(--ds-ink)" strokeWidth="1" />
                    <line x1={xs(0)} x2={xs(0)} y1={PY} y2={H - PY}
                        stroke="var(--ds-ink)" strokeWidth="1" />

                    {/* Tick labels */}
                    {xTicks.map((t) => (
                        <text key={`tx${t}`} x={xs(t)} y={ys(0) + 16}
                            fontSize="11" fontFamily="var(--ds-font-mono)"
                            textAnchor="middle" fill="var(--ds-faint)">{t}</text>
                    ))}
                    {yTicks.filter((t) => t > 0).map((t) => (
                        <text key={`tl${t}`} x={xs(0) - 8} y={ys(t) + 4}
                            fontSize="11" fontFamily="var(--ds-font-mono)"
                            textAnchor="end" fill="var(--ds-faint)">{t}</text>
                    ))}

                    {/* Axis labels */}
                    <text x={W - PX + 6} y={ys(0) + 4} fontSize="13"
                        fontStyle="italic" fontFamily="var(--ds-font-serif)" fill="var(--ds-ink)">w</text>
                    <text x={xs(0) + 8} y={PY + 4} fontSize="13"
                        fontStyle="italic" fontFamily="var(--ds-font-serif)" fill="var(--ds-ink)">ℒ(w)</text>

                    {/* Loss curve */}
                    <polyline points={curvePts} fill="none"
                        stroke="var(--ds-accent)" strokeWidth="1.6" />

                    {/* Tangent line at current w */}
                    <line
                        x1={xs(tx0)} y1={ys(ty0)}
                        x2={xs(tx1)} y2={ys(ty1)}
                        stroke="var(--ds-warm)" strokeWidth="1.25" strokeDasharray="4 4"
                    />

                    {/* Trail of prior iterates */}
                    {trail.map((p, i) => (
                        <circle key={`trail-${i}`}
                            cx={xs(p.weight)} cy={ys(p.loss)} r="3"
                            fill="var(--ds-warm)"
                            opacity={0.3 + 0.5 * (i / Math.max(1, trail.length))}
                        />
                    ))}

                    {/* Active ball — animated via ref while running */}
                    <g>
                        <circle
                            ref={ballRef}
                            cx={xs(w)} cy={ys(lossHere)} r="6.5"
                            fill="var(--ds-paper)"
                            stroke="var(--ds-warm)" strokeWidth="2"
                        />
                    </g>

                    {/* Annotation */}
                    {current.iteration > 0 && (
                        <g>
                            <line x1={xs(w) + 10} y1={ys(lossHere) - 14}
                                x2={xs(w) + 64} y2={ys(lossHere) - 44}
                                stroke="var(--ds-faint)" strokeWidth="0.8" />
                            <text x={xs(w) + 70} y={ys(lossHere) - 42}
                                fontSize="12" fontStyle="italic"
                                fontFamily="var(--ds-font-serif)" fill="var(--ds-faint)">
                                step {current.iteration} · w = {w.toFixed(3)}
                            </text>
                        </g>
                    )}
                </svg>

                <div className="gd-controls-strip">
                    <button className="ds-btn primary" onClick={runDescent} disabled={isRunning}>
                        {isRunning ? 'Running…' : 'Run descent →'}
                    </button>
                    <button className="ds-btn ghost" onClick={reset} disabled={isRunning}>
                        Reset
                    </button>
                </div>
            </div>
        </Figure>
    );
}
