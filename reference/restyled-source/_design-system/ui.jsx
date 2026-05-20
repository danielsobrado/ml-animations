// ============================================================================
// ui.jsx — React primitives for the Distill design system.
// All visual styling lives in distill.css; these components just compose markup.
// ============================================================================
import React from 'react';
import Eq from './Eq';

/* ----- Page ----------------------------------------------------------- */
export function Page({ children }) {
    return <div className="ds-page"><div className="ds-shell">{children}</div></div>;
}

/* ----- Header --------------------------------------------------------- */
export function Header({ eyebrow, title, subtitle, right }) {
    return (
        <header className="ds-header">
            <div className="ds-eyebrow">
                {eyebrow.map((part, i) => (
                    <React.Fragment key={i}>
                        <span>{part}</span>
                        {i < eyebrow.length - 1 && <span className="sep">·</span>}
                    </React.Fragment>
                ))}
                {right && <span className="right">{right}</span>}
            </div>
            <h1 className="ds-title">{title}</h1>
            {subtitle && <p className="ds-subtitle">{subtitle}</p>}
        </header>
    );
}

/* ----- Equation strip ------------------------------------------------- */
export function EquationStrip({ label, tex, meta }) {
    return (
        <div className="ds-eq-strip">
            <div className="label">{label}</div>
            <div className="eq"><Eq tex={tex} /></div>
            {meta && <div className="meta">{meta}</div>}
        </div>
    );
}

/* ----- Figure (panel with head + body + figcaption) ------------------- */
export function Figure({ label, title, right, children, caption }) {
    return (
        <figure className="ds-figure">
            <div className="ds-panel">
                {(label || title || right) && (
                    <div className="ds-panel-head">
                        <span>{label}{title && <span style={{ marginLeft: 8, color: 'var(--ds-ink)', textTransform: 'none', letterSpacing: 0, fontFamily: 'var(--ds-font-serif)', fontStyle: 'italic', fontSize: 13 }}> — {title}</span>}</span>
                        {right && <span>{right}</span>}
                    </div>
                )}
                <div>{children}</div>
            </div>
            {caption && <figcaption className="ds-figcaption">{caption}</figcaption>}
        </figure>
    );
}

/* ----- Readouts (tabular metric stack) -------------------------------- */
export function Readouts({ rows }) {
    // rows: [{ label, tex, value }]
    return (
        <div className="ds-readouts">
            {rows.map((r) => (
                <div key={r.label} className="ds-readout-row">
                    <div className="ds-readout-label">{r.label}</div>
                    <div className="ds-readout-tex">{r.tex && <Eq tex={r.tex} />}</div>
                    <div className="ds-readout-val">{r.value}</div>
                </div>
            ))}
        </div>
    );
}

/* ----- Aside (margin notes / controls column) ------------------------- */
export function Aside({ heading, children }) {
    return (
        <aside className="ds-aside">
            {heading && <div className="ds-aside-head">{heading}</div>}
            {children}
        </aside>
    );
}

/* ----- ParamSlider ---------------------------------------------------- */
export function ParamSlider({
    label, tex, value, min, max, step,
    onChange,
    format = (v) => v.toFixed(2),
    hint, hintTone = 'ok',
}) {
    return (
        <div className="ds-slider">
            <div className="ds-slider-head">
                <div className="ds-slider-label">
                    {label}
                    {tex && <span className="tex"><Eq tex={tex} /></span>}
                </div>
                <div className="ds-slider-value">{format(value)}</div>
            </div>
            <input
                className="ds-range"
                type="range" min={min} max={max} step={step} value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
            />
            {hint && <div className={`ds-slider-hint ${hintTone}`}>{hint}</div>}
        </div>
    );
}

/* ----- Buttons -------------------------------------------------------- */
export function Btn({ variant = 'primary', children, ...rest }) {
    return <button className={`ds-btn ${variant}`} {...rest}>{children}</button>;
}
export function BtnRow({ children }) {
    return <div className="ds-btn-row">{children}</div>;
}

/* ----- Re-export Eq for convenience ----------------------------------- */
export { Eq };
