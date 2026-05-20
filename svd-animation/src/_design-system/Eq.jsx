// ============================================================================
// Eq — KaTeX wrapper. Use <Eq tex="\\alpha" /> for inline, <Eq tex="..." block />
// for display mode. KaTeX is loaded as a dep (npm i katex).
// ============================================================================
import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export default function Eq({ tex, block = false, className }) {
    const ref = useRef(null);
    useEffect(() => {
        if (!ref.current) return;
        try {
            katex.render(tex, ref.current, {
                throwOnError: false,
                displayMode: block,
                output: 'html',
            });
        } catch (e) {
            // fall through with raw tex visible
        }
    }, [tex, block]);
    return <span ref={ref} className={className} />;
}
