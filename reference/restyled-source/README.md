# ML Animations — Distill design system

A unified visual system for the ML Animations catalog. Distill.pub-inspired:
warm paper background, Source Serif headings, hairline rules, single ink-blue
accent, KaTeX math throughout.

## What's in this folder

```
_design-system/
  distill.css      Tokens + base + component styles (CSS only, no JS deps)
  Eq.jsx           <Eq tex="..."/> KaTeX wrapper
  ui.jsx           Page / Header / EquationStrip / Figure / Readouts / Aside / ParamSlider / Btn
```

Plus a fully restyled `gradient-descent-animation/` as the reference port.

## One-time setup (per animation project)

1. **Copy the module.** From the repo root, copy `_design-system/` into the
   animation's `src/` folder:

   ```bash
   cp -r restyled-source/_design-system  <animation>/src/_design-system
   ```

2. **Add KaTeX as a dependency.** In the animation's `package.json`:

   ```json
   "dependencies": {
     "katex": "^0.16.11",
     ...
   }
   ```

   Then `npm install` from that animation's folder.

3. **Add Google Fonts to `index.html`.** Inside `<head>`:

   ```html
   <link rel="preconnect" href="https://fonts.googleapis.com" />
   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
   <link
     href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap"
     rel="stylesheet" />
   ```

4. **Import the stylesheet from `src/index.css`** (alongside the Tailwind
   directives — Tailwind still works, our classes are namespaced `ds-*`):

   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;

   @import './_design-system/distill.css';
   ```

## The shape of every page

```jsx
import {
  Page, Header, EquationStrip, Figure,
  Readouts, Aside, ParamSlider, Btn, BtnRow, Eq,
} from './_design-system/ui';

export default function App() {
  return (
    <Page>
      <Header
        eyebrow={['Chapter NN', 'Topic', '§ N.N']}
        title="Lesson title"
        subtitle={<>Italic serif subtitle with <Eq tex="\\alpha" /> inline math.</>}
      />

      <EquationStrip
        label="Definition"
        tex="\\text{the headline equation goes here}"
        meta={<>with <Eq tex="\\text{side condition}" /></>}
      />

      <div className="page-body">
        <div className="main-col">
          {/* Figures, each wrapped in <Figure label title caption> */}
        </div>
        <aside className="side-col">
          <Aside heading="Controls">
            {/* ParamSliders, narration, tips */}
          </Aside>
        </aside>
      </div>
    </Page>
  );
}
```

The grid utilities (`.page-body`, `.main-col`, `.readouts-and-history`,
`.side-col`) live in **each animation's** `src/index.css`. Copy them from
`gradient-descent-animation/src/index.css` and tweak per layout.

## Porting checklist (find/replace patterns)

For each remaining animation, work through these in order:

### 1. Strip the dark theme & hot colors

Most animations use `bg-gradient-to-br from-slate-950 via-X-950 to-Y-950` or
similar. Delete the whole outer wrapper and replace with `<Page>`.

| Find | Replace |
|---|---|
| `<div className="min-h-screen bg-gradient-to-br ...">` | `<Page>` |
| `<div className="min-h-screen bg-gray-100 p-4">` | `<Page>` |
| `bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700` | (remove — `<Figure>` provides chrome) |
| `text-transparent bg-clip-text bg-gradient-to-r ...` | (remove — `<Header title>` styles) |

### 2. Replace headers

| Find | Replace |
|---|---|
| `<h1 className="text-5xl font-extrabold ...">{title}</h1>` | `<Header eyebrow={[...]} title={title} subtitle={...} />` |
| `<p className="text-slate-300 text-lg">{subtitle}</p>` | (folded into `<Header subtitle>`) |

### 3. Replace tab/nav bars

The catalog has lots of `flex flex-wrap justify-center gap-3 mb-8` tab bars
with gradient pills. Replace with a simple eyebrow-style nav row — see the
shared pattern below (drop into `ui.jsx` if you want it as a primitive):

```jsx
<nav className="ds-tabs">
  {tabs.map((t) => (
    <button
      key={t.id}
      className={`ds-tab ${active === t.id ? 'active' : ''}`}
      onClick={() => setActive(t.id)}
    >
      <span className="num">{String(t.idx).padStart(2, '0')}</span>
      <span className="label">{t.label}</span>
    </button>
  ))}
</nav>
```

Then add to `distill.css`:

```css
.ds-tabs { display: flex; gap: 0; border-bottom: var(--ds-border); margin-bottom: var(--ds-gap-3); }
.ds-tab {
  background: transparent; border: none; cursor: pointer;
  padding: 12px 22px 14px;
  font-family: var(--ds-font-sans); font-size: var(--ds-text-sm);
  color: var(--ds-faint);
  border-bottom: 2px solid transparent;
  display: flex; align-items: baseline; gap: 8px;
}
.ds-tab .num { font-family: var(--ds-font-mono); font-size: 11px; color: var(--ds-mute); }
.ds-tab:hover { color: var(--ds-ink); }
.ds-tab.active { color: var(--ds-ink); border-bottom-color: var(--ds-accent); }
.ds-tab.active .num { color: var(--ds-accent); }
```

### 4. Replace panel chrome

| Find | Replace |
|---|---|
| `bg-gray-50 rounded-xl shadow-lg overflow-hidden` | wrap content in `<Figure label="Figure N" title="..." caption={...}>` |
| `bg-blue-50 p-4 rounded-lg border border-blue-200` (tip boxes) | move into `<Aside>` paragraph, or use `ds-eq-strip` styling |

### 5. Replace buttons

| Find | Replace |
|---|---|
| `bg-green-500 hover:bg-green-600 ... rounded-lg` (Play/Run) | `<Btn variant="primary">Run →</Btn>` |
| `bg-red-500 hover:bg-red-600 ... rounded-lg` (Reset) | `<Btn variant="ghost">Reset</Btn>` |
| `bg-blue-500 hover:bg-blue-600 ... rounded-lg` (Next/Prev) | `<Btn variant="ghost">Next</Btn>` |

Wrap pairs in `<BtnRow>`.

### 6. Replace sliders

| Find | Replace |
|---|---|
| `<input type="range" className="... h-2 bg-gray-200 rounded-lg ..."/>` + label + value display | `<ParamSlider label tex value min max step onChange format hint hintTone />` |

### 7. Replace inline math literals

Search each panel for `α`, `∇`, `ℒ`, `→`, manually-built fractions, `√`, `²`,
etc. Wrap in `<Eq tex="..."/>`. Examples:

```jsx
// Before
"Learning Rate (α):"
"Loss: <span>{(w*w).toFixed(3)}</span>"
"−ln(p) = 0.36"

// After
<>Learning rate <Eq tex="\alpha" /></>
<>Loss <Eq tex="\mathcal{L}(w) = " /> <strong>{(w*w).toFixed(3)}</strong></>
<Eq tex="-\ln(p) = 0.36" />
```

### 8. Replace the colour palette in viz code

The animations hard-code colours in `COLORS` objects at the top of each panel.
Swap to the system palette:

```js
// Before
const COLORS = {
  curve: 0x7030a0,      // Purple
  ball: 0x5b9bd5,       // Blue
  gradient: 0xed7d31,   // Orange
  optimal: 0x70ad47,    // Green
  bg: 0xffffff,
};

// After
const COLORS = {
  curve:    0x264273,   // var(--ds-accent)
  ball:     0xa85a3a,   // var(--ds-warm)
  gradient: 0xa85a3a,   // var(--ds-warm) — same family
  optimal:  0x3a6a3a,   // var(--ds-ok), used sparingly
  bg:       0xfbf8f1,   // var(--ds-paper)
  ink:      0x1a1a1a,
  grid:     0xece6d3,
  faint:    0x4a4a4a,
};
```

For HTML/CSS contexts use the `var(--ds-*)` names directly.

### 9. Recommended: lift Three.js 2D plots to SVG

The `*Panel.jsx` files that use Three.js with an OrthographicCamera to draw
2D parabolas / 1D vectors are paying for it: WebGL `LineBasicMaterial` caps
the line at 1px on most platforms, making curves look frail. The Gradient
Descent restyle replaces the Three.js layer with a plain `<svg>` — the
animation logic (GSAP tween) is unchanged, just the canvas was swapped for
SVG attributes via a ref. Apply the same lift to:

- `cross-entropy-animation/CrossEntropyPanel.jsx`
- `entropy-animation/EntropyPanel.jsx`
- `cosine-similarity-animation/*`
- `eigenvalue-animation/*`
- any panel whose Three.js is just `OrthographicCamera + lines + sprites`

Keep Three.js for actually-3D panels (joint attention, conv2d, dit-transformer,
gpt2-comprehensive). Update those to use the new palette.

### 10. Pedagogical tightening

While you're in each file, two small content cleanups:

- **Slider hints** should say *what's about to happen at this value*, not just
  "Good" / "Too slow". Use the `hint` + `hintTone` props on `ParamSlider`.
- **Figure captions** should reference dynamic values via `Eq tex={\`...\`}`
  rather than hard-coding numbers in prose. The original Gradient Descent
  caption hard-coded `w = 0.732` independent of the actual data — the
  restyle pulls the value from state.

## Tokens reference

CSS variables — use these directly in any inline style or component CSS:

```
--ds-paper   #fbf8f1   page background
--ds-panel   #fefcf7   figure background
--ds-ink     #1a1a1a   primary text
--ds-faint   #4a4a4a   secondary text
--ds-mute    #7a7a78   tertiary text
--ds-rule    #d9d2c0   hairline borders
--ds-grid    #ece6d3   chart gridlines
--ds-accent  #264273   primary accent (loss curves, sliders fill)
--ds-warm    #a85a3a   secondary accent (current iterate, tangents)
--ds-ok      #3a6a3a   positive status (sparing)
--ds-warn    #a85a3a   warning status
--ds-bad     #8a1d1d   error status

--ds-font-serif  Source Serif 4 — headings, captions, body prose
--ds-font-sans   Inter — UI labels, buttons
--ds-font-mono   JetBrains Mono — numbers, tags, eyebrows
```

## Suggested porting order

Tackle in this order — earlier animations exercise more of the system, so
later ones become near-mechanical:

1. **gradient-descent** ✓ (reference port)
2. **cross-entropy** — exercises step navigation, vector cells, formula chain
3. **gradient-descent**, **conditional-probability**, **entropy** — pure 2D plots
4. **embeddings**, **cosine-similarity**, **fasttext**, **glove** — vector viz
5. **conv2d**, **conv-relu** — grid heatmaps
6. **attention-mechanism**, **bert**, **gpt2-comprehensive** — multi-tab transformer pages
7. Everything else
