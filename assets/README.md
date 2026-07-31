# convsearch brand assets

## Concept

The mark is two rounded tiles stacked like archived pages, with the front
tile punched through by a perfectly round porthole — a search lens looking
back into your own history. The hole is real transparency (an SVG mask),
not a drawn circle, so the mark always shows whatever is behind it instead
of needing a background color to "complete" the shape. It reads as
**layered history you can look into**, which is the whole pitch of
convsearch: your ChatGPT conversations, archived locally, searchable by
you and no one else.

## Files

- `logo.svg` — the mark alone, square, transparent background, `viewBox="0 0 100 100"`.
- `logo-wordmark.svg` — the same mark plus "convsearch" set in a system font
  stack (`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
  Arial, sans-serif`), `viewBox="0 0 560 100"`. No external or embedded font
  files — it renders anywhere.
- `../extension/icons/icon{16,32,48,128}.png` — MV3 toolbar/store icons
  rasterized from `logo.svg`, transparent RGBA PNGs at the declared pixel
  dimensions.

## Color

Both SVGs use `fill="currentColor"` throughout — there are no hardcoded
hex values in the source. Set the CSS `color` property on a parent element
(or `style="color: ..."` directly on the `<svg>`) to theme the mark:

```html
<!-- on a light surface -->
<img src="logo.svg" style="color: #1E1B2E">

<!-- on a dark surface -->
<img src="logo.svg" style="color: #F5F3FF">
```

(Note: `<img>` tags can't apply `currentColor` to SVG content across all
browsers — for guaranteed color inheritance, inline the SVG or reference it
with `<object>`/`<svg><use>`, or bake a fixed color into a copy as the
raster icons below do.)

Where a fixed, non-inherited color is required (raster icons, marketing
decks, anywhere `currentColor` can't reach), use this two-tone palette:

| Name  | Hex       | Use                                              |
|-------|-----------|---------------------------------------------------|
| Ink   | `#1E1B2E` | Primary mark color on light or neutral backgrounds |
| Paper | `#F5F3FF` | Inverse mark color on dark backgrounds             |

The extension icons in `extension/icons/` are baked with **Ink** (`#1E1B2E`)
on a transparent background, which stays legible against both Chrome's
light and dark toolbar chrome.

## Clear space and minimum size

- Keep clear space around the mark equal to at least the width of the back
  tile's offset (roughly 10% of the mark's width) on every side. Don't crop
  the mark tighter than its own bounding box.
- Minimum size: 16px square for the mark alone (it was designed and tested
  down to a real 16×16 raster favicon), 20px tall for the wordmark lockup.
  Below that, the porthole and the tile stagger stop resolving — don't go
  smaller.
- The mark and wordmark are both bold, flat shapes with no thin strokes and
  no fine internal detail by design, specifically so they survive
  downscaling to a favicon. Don't add detail that would break that.

## What not to do

- Don't recolor the two tiles differently or add a third color — the mark
  is deliberately monochrome (one color at two opacities) so it works with
  `currentColor` and never clashes with a surrounding palette.
- Don't fill in the porthole (the circular cutout). It must stay
  transparent — that's the "search lens into your archive" idea, and a
  filled-in circle reads as a plain icon with a dot, not a lens.
- Don't add a drop shadow, bevel, gradient fill, or outline stroke to the
  tiles. The flat silhouette is what keeps it legible at 16px.
- Don't stretch the mark non-uniformly or rotate it. It's designed as a
  square aligned to its own corners.
- Don't set text inside the mark itself, and don't typeset "convsearch" in
  any font other than a system UI sans-serif stack (no display/script
  fonts, no custom webfont) — the wordmark should render instantly with no
  font loading, anywhere, offline.
- Don't put the wordmark lockup on a background where the contrast between
  `currentColor` and the surface falls below roughly 4.5:1 (i.e. don't use
  Ink-on-dark or Paper-on-light).

## Regenerating the extension icons

`extension/icons/icon{16,32,48,128}.png` were rasterized from `logo.svg`
using the Playwright Chromium binary already vendored for this repo's
end-to-end tests (`node_modules/playwright`), via a small one-off script
that renders the SVG in a headless page and captures a screenshot with a
transparent background override through the CDP `Page.captureScreenshot` /
`Emulation.setDefaultBackgroundColorOverride` calls (Playwright's own
`page.screenshot({ omitBackground: true })` errors out on the chromium
build available in this environment; calling `Page.captureScreenshot`
once, ignoring the resulting error, then setting the background override
and capturing again reliably produces a true alpha channel). No new
dependency was added to the project to do this — Pillow and pip were not
available in `.venv`, so the existing Playwright/Chromium install already
used for `tests-e2e` was reused instead. If you change `logo.svg`, rerun
that pipeline to refresh the four PNGs; there is no dependency on it being
committed anywhere in this repo, since the source SVG is the source of
truth.
