# ClaimGuard-CXR Brand Palette

Custom medical-grade palette. **Do not use default Tailwind indigo/blue/purple** — the CLAUDE.md rules forbid generic Tailwind palette colors as primary.

## Primary — Deep Teal

Anchors the brand. Medical, trustworthy, calm. Distinct from generic "tech blue".

| Token | Hex | Usage |
|---|---|---|
| `teal-950` | `#042F2E` | Darkest surfaces, deep backgrounds |
| `teal-900` | `#134E4A` | Dark-section backgrounds |
| `teal-800` | `#115E59` | Hover states on teal elements |
| `teal-700` | `#0F766E` | **PRIMARY** — brand color, primary CTAs, headings accent |
| `teal-600` | `#0D9488` | Primary button hover |
| `teal-500` | `#14B8A6` | Highlights, links on dark surfaces |
| `teal-400` | `#2DD4BF` | Bright accents, data viz |
| `teal-200` | `#99F6E4` | Tint backgrounds, soft emphasis |
| `teal-100` | `#CCFBF1` | Faint backgrounds, tags |
| `teal-50`  | `#F0FDFA` | Subtle section tints |

## Semantic — Triage States (match the demo output)

These mirror the RED/YELLOW/GREEN color coding from the live verifier demo.

| Token | Hex | Usage |
|---|---|---|
| `safe` (sage) | `#65A30D` | GREEN — verified claims |
| `safe-light` | `#D9F99D` | GREEN background tint |
| `warn` (amber) | `#D97706` | YELLOW — needs-review claims |
| `warn-light` | `#FEF3C7` | YELLOW background tint |
| `danger` (coral) | `#DC2626` | RED — contradicted / hallucinated claims |
| `danger-light` | `#FEE2E2` | RED background tint |

Intentionally avoiding pure `#22C55E` / `#EF4444` / `#EAB308` (which are the exact Tailwind-default values used in `demo/app.py`) — the website palette is a more desaturated, editorial sibling of the demo palette so the marketing site reads as a polished product page rather than a dev tool UI.

## Neutrals — Warm Gray (stone family)

Warmer than default Tailwind `neutral` or `slate`. Reads more editorial, less corporate.

| Token | Hex | Usage |
|---|---|---|
| `stone-950` | `#0C0A09` | Near-black text, dark section backgrounds |
| `stone-900` | `#1C1917` | Body text on light backgrounds |
| `stone-800` | `#292524` | Secondary text |
| `stone-700` | `#44403C` | Tertiary text, icon defaults |
| `stone-600` | `#57534E` | Muted text |
| `stone-500` | `#78716C` | Placeholder text, subtle borders |
| `stone-400` | `#A8A29E` | Disabled text |
| `stone-300` | `#D6D3D1` | Borders, dividers |
| `stone-200` | `#E7E5E4` | Card borders, rules |
| `stone-100` | `#F5F5F4` | Elevated surface backgrounds |
| `stone-50`  | `#FAFAF9` | Page background |

## Shadows — Teal-tinted (no flat Tailwind shadows)

```css
/* Elevation 1 — cards, subtle lift */
box-shadow:
  0 1px 2px 0 rgba(15, 118, 110, 0.04),
  0 2px 4px -2px rgba(15, 118, 110, 0.04);

/* Elevation 2 — popovers, stat cards */
box-shadow:
  0 4px 8px -2px rgba(15, 118, 110, 0.06),
  0 8px 24px -6px rgba(15, 118, 110, 0.08);

/* Elevation 3 — floating hero elements */
box-shadow:
  0 16px 32px -8px rgba(15, 118, 110, 0.10),
  0 32px 64px -16px rgba(15, 118, 110, 0.12),
  inset 0 1px 0 0 rgba(255, 255, 255, 0.5);
```

## Typography

**Important:** The `frontend-design` skill explicitly bans Inter, Roboto, Arial, and system fonts as "generic AI-slop". We use distinctive alternatives.

- **Display / headings:** `Fraunces` (variable serif, Google Fonts) — `SOFT 100, opsz 144, wght 400-600`, tracking `-0.03em` on large headings, generous `-0.02em` on subheads. Fraunces is characterful, editorial, and avoids generic serif clichés.
- **Body / UI:** `Manrope` (variable humanist-geometric sans, Google Fonts) — `wght 400-600`, line-height `1.7` on long-form paragraphs, `-0.01em` tracking. Manrope has distinctive round counters and a warm, engineered feel — neither generic like Inter nor overused like Space Grotesk.
- **Monospace / code / numbers:** `JetBrains Mono` (variable, Google Fonts) — used for stat card numbers and code blocks for a technical-precise feel against the softer serif headings.

## Aesthetic Direction

**Commitment:** Editorial medical-futurism. Serif editorial authority (Fraunces) meets clean engineered sans (Manrope) meets technical monospace (JetBrains Mono). Warm off-white page background with deep teal accents. Grain-noise overlay on dark sections for analog depth. Asymmetric, generously spaced hero. Oversized numeric stat cards. Horizontal rules as editorial dividers, not flat borders.

**Tone vocabulary:** Trustworthy. Precise. Confident. Editorial. Never cheerful, never corporate-generic, never dashboard-busy.

## Texture — SVG Noise Overlay

Use at ~3% opacity on dark sections and feature cards to add depth and break up flat fills. Inline SVG definition:

```html
<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0">
  <filter id="noise">
    <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" />
    <feColorMatrix type="saturate" values="0" />
  </filter>
</svg>
```

Apply via a fixed pseudo-element with `filter: url(#noise)` and `opacity: 0.03`.

## Gradients — Layered Radial

Per CLAUDE.md guardrails, use multiple layered radial gradients, not single linear gradients. Example hero backdrop:

```css
background:
  radial-gradient(at 20% 10%, rgba(15, 118, 110, 0.14) 0%, transparent 50%),
  radial-gradient(at 80% 20%, rgba(45, 212, 191, 0.08) 0%, transparent 45%),
  radial-gradient(at 50% 90%, rgba(4, 47, 46, 0.10) 0%, transparent 55%),
  #FAFAF9;
```
