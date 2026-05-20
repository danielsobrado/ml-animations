// Shared math + chart helpers for all three variants.
// Babel-transpiled scripts don't share scope, so we export to window.

const PARABOLA = (w) => w * w;

// Simulate a gradient descent run for the history chart.
// Mid-trajectory snapshot: starting at w=3.0, lr=0.25, after 3 steps.
function simulate(start = 3.0, lr = 0.25, steps = 12) {
  const out = [{ i: 0, w: start, L: PARABOLA(start) }];
  let w = start;
  for (let i = 1; i <= steps; i++) {
    const grad = 2 * w;
    w = w - lr * grad;
    out.push({ i, w, L: PARABOLA(w) });
  }
  return out;
}

// Render a KaTeX expression into a span, returning the JSX.
function K({ tex, displayMode = false, color }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !window.katex) return;
    try {
      window.katex.render(tex, ref.current, {
        throwOnError: false,
        displayMode,
        output: 'html',
      });
    } catch (e) {}
  }, [tex, displayMode]);
  return <span ref={ref} style={color ? { color } : undefined} />;
}

// Re-render KaTeX once the script loads (since it loads `defer`).
function useKatexReady() {
  const [, force] = React.useReducer((x) => x + 1, 0);
  React.useEffect(() => {
    if (window.katex) return;
    const id = setInterval(() => {
      if (window.katex) {
        clearInterval(id);
        force();
      }
    }, 50);
    return () => clearInterval(id);
  }, []);
}

Object.assign(window, { PARABOLA, simulate, K, useKatexReady });
