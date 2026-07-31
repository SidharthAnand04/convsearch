// render-launch-visuals.mjs — LAUNCH-VISUALS renderer.
//
// Renders on-brand launch imagery (og card, GitHub banner, Product Hunt gallery)
// by laying out HTML/CSS and screenshotting it with Playwright/Chromium at exact
// pixel sizes. No server, no network — everything is inlined (the real wordmark
// from assets/logo-wordmark.svg, the real side-panel screenshot as a data URI).
//
//   node scripts/render-launch-visuals.mjs
//
// Owned output paths:
//   site/media/og-card.png                 1200x630
//   docs/launch/assets/github-banner.png   1280x640
//   docs/launch/assets/ph-gallery-01.png   1270x760
//   docs/launch/assets/ph-gallery-02.png   1270x760
//   docs/launch/assets/ph-gallery-03.png   1270x760
//   docs/launch/assets/ph-gallery-04.png   1270x760

import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");

const SITE_MEDIA = path.join(REPO, "site", "media");
const LAUNCH_ASSETS = path.join(REPO, "docs", "launch", "assets");
fs.mkdirSync(SITE_MEDIA, { recursive: true });
fs.mkdirSync(LAUNCH_ASSETS, { recursive: true });

/* ---- inlined real assets -------------------------------------------------- */

const panelPng = fs.readFileSync(path.join(REPO, "docs", "screenshots", "panel.png"));
const PANEL_URI = `data:image/png;base64,${panelPng.toString("base64")}`;

// The mark geometry, identical to assets/logo.svg / logo-wordmark.svg (porthole mask).
const MARK = (id) => `
<svg viewBox="0 0 100 100" class="mark" aria-hidden="true">
  <defs><mask id="pm-${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">
    <rect x="0" y="0" width="100" height="100" fill="#fff"/>
    <circle cx="58" cy="58" r="17" fill="#000"/>
  </mask></defs>
  <rect x="9" y="9" width="58" height="58" rx="16" fill="currentColor" opacity="0.4"/>
  <rect x="29" y="29" width="58" height="58" rx="16" fill="currentColor" mask="url(#pm-${id})"/>
</svg>`;

// Full wordmark lockup (mark + type), themable via currentColor.
const WORDMARK = (id) => `
<svg viewBox="0 0 560 100" class="wordmark" aria-label="convsearch">
  <defs><mask id="wm-${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">
    <rect x="0" y="0" width="100" height="100" fill="#fff"/>
    <circle cx="58" cy="58" r="17" fill="#000"/>
  </mask></defs>
  <rect x="9" y="9" width="58" height="58" rx="16" fill="currentColor" opacity="0.4"/>
  <rect x="29" y="29" width="58" height="58" rx="16" fill="currentColor" mask="url(#wm-${id})"/>
  <text x="122" y="66" fill="currentColor"
    font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    font-size="52" font-weight="600" letter-spacing="-0.5">convsearch</text>
</svg>`;

/* ---- shared design system ------------------------------------------------- */

const BASE = `
  --paper:#070417; --panel:#141031; --sunken:#0d0a24;
  --primary:#9d8cec; --secondary:#8e1884; --accent:#de3a90;
  --accent-2:#b9abf5; --ink:#dad4f8;
  --muted:rgba(218,212,248,0.64); --faint:rgba(218,212,248,0.42);
  --line:rgba(157,140,236,0.16); --line-2:rgba(157,140,236,0.34);
  --accent-soft:rgba(157,140,236,0.14);
  --dot-online:#59d499;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --mono:'SFMono-Regular',ui-monospace,'JetBrains Mono','Cascadia Code',Menlo,Consolas,monospace;
`;

function doc(w, h, body, extraCss = "") {
  return `<!doctype html><html><head><meta charset="utf-8"><style>
  *{margin:0;padding:0;box-sizing:border-box;}
  html,body{width:${w}px;height:${h}px;}
  body{ ${BASE}
    font-family:var(--sans); color:var(--ink);
    background:var(--paper);
    -webkit-font-smoothing:antialiased; text-rendering:geometricPrecision;
    overflow:hidden; position:relative;
  }
  /* atmospheric ground: layered purple glows + faint grid */
  .stage{position:absolute; inset:0; overflow:hidden;}
  .stage::before{content:""; position:absolute; inset:0;
    background:
      radial-gradient(120% 90% at 82% -12%, rgba(222,58,144,0.20), transparent 55%),
      radial-gradient(120% 120% at 8% 108%, rgba(142,24,132,0.28), transparent 52%),
      radial-gradient(90% 90% at 50% 0%, rgba(157,140,236,0.12), transparent 60%);
  }
  .grid{position:absolute; inset:0; opacity:0.5;
    background-image:
      linear-gradient(rgba(157,140,236,0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(157,140,236,0.05) 1px, transparent 1px);
    background-size:44px 44px; mask-image:radial-gradient(120% 100% at 30% 40%,#000,transparent 78%);}
  .content{position:absolute; inset:0;}

  .mark{display:block;}
  .wordmark{display:block; color:var(--ink);}
  .lockup .wordmark{height:var(--wm,34px); width:auto;}

  .eyebrow{font-family:var(--mono); text-transform:uppercase; letter-spacing:0.28em;
    font-size:13px; color:var(--primary); display:flex; align-items:center; gap:12px;}
  .eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);}

  h1{font-weight:700; letter-spacing:-0.022em; line-height:1.02;}
  .grad{background:linear-gradient(100deg,var(--accent-2) 0%,#c9a7ff 30%,var(--accent) 70%,var(--secondary) 118%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;}
  .thin{color:var(--muted); font-weight:300;}

  .chips{display:flex; gap:10px; flex-wrap:wrap;}
  .chip{font-family:var(--mono); font-size:12.5px; letter-spacing:0.04em;
    padding:7px 13px; border-radius:999px; border:1px solid var(--line-2);
    background:rgba(157,140,236,0.07); color:var(--ink); white-space:nowrap;}
  .chip.hot{border-color:rgba(222,58,144,0.5); background:rgba(222,58,144,0.12); color:#f7b7d6;}

  .url{font-family:var(--mono); color:var(--muted); letter-spacing:0.02em; font-size:16px;}
  .url b{color:var(--ink); font-weight:600;}
  .lic{font-family:var(--mono); font-size:12px; letter-spacing:0.16em; color:var(--faint); text-transform:uppercase;}

  /* device frame for the panel screenshot */
  .device{position:relative; border-radius:20px; padding:10px;
    background:linear-gradient(160deg, rgba(157,140,236,0.22), rgba(142,24,132,0.10));
    border:1px solid var(--line-2);
    box-shadow:0 40px 90px -30px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.02) inset;}
  .device .screen{border-radius:12px; overflow:hidden; background:var(--paper);
    border:1px solid rgba(0,0,0,0.5);}
  .device img{display:block;}
  .glowline{position:absolute; border-radius:inherit; inset:0;
    box-shadow:0 0 60px -10px rgba(222,58,144,0.5); pointer-events:none;}

  ${extraCss}
  </style></head><body><div class="stage"><div class="grid"></div></div><div class="content">${body}</div></body></html>`;
}

/* ---------------------------------------------------------------------------
 * 1) OG CARD — 1200x630
 * ------------------------------------------------------------------------- */
function ogCard() {
  const css = `
  .wrap{position:absolute; inset:0; display:grid; grid-template-columns:1fr 430px;}
  .left{padding:64px 20px 60px 72px; display:flex; flex-direction:column; justify-content:space-between;}
  .left h1{font-size:57px; margin:26px 0 22px;}
  .left .tag{font-size:20px; line-height:1.5; color:var(--muted); max-width:20ch;}
  .footer{display:flex; align-items:center; justify-content:space-between; gap:20px;}
  .lockup{--wm:36px;}
  .right{position:relative; overflow:hidden;}
  .right .device{position:absolute; top:-58px; left:34px; width:520px; transform:rotate(3.2deg);}
  .right .device img{width:100%;}
  .fade{position:absolute; inset:0; z-index:2; pointer-events:none;
    background:linear-gradient(90deg, var(--paper) 0%, transparent 20%),
              linear-gradient(0deg, rgba(7,4,23,0.55), transparent 40%);}
  `;
  const body = `<div class="wrap">
    <div class="left">
      <div>
        <div class="lockup">${WORDMARK("og")}</div>
        <div class="eyebrow" style="margin-top:30px;"><span class="dot"></span>Local-first · Chrome side panel</div>
        <h1>Search your <span class="grad">ChatGPT history</span> like you own it.</h1>
        <div class="tag">Hybrid keyword + semantic search, grounded cited answers, and explainable ranking — 100% on your machine.</div>
      </div>
      <div class="footer">
        <div class="url"><b>convsearch</b>.vercel.app</div>
        <div class="lic">Apache-2.0</div>
      </div>
    </div>
    <div class="right">
      <div class="device"><div class="screen"><img src="${PANEL_URI}"></div><div class="glowline"></div></div>
      <div class="fade"></div>
    </div>
  </div>`;
  return doc(1200, 630, body, css);
}

/* ---------------------------------------------------------------------------
 * 2) GITHUB BANNER — 1280x640
 * ------------------------------------------------------------------------- */
function githubBanner() {
  const css = `
  .wrap{position:absolute; inset:0; display:grid; grid-template-columns:1fr 470px;}
  .left{padding:74px 24px 66px 84px; display:flex; flex-direction:column; justify-content:space-between;}
  .lockup{--wm:44px;}
  .left h1{font-size:52px; margin:30px 0 24px; max-width:15ch;}
  .left .tag{font-size:19px; line-height:1.55; color:var(--muted); max-width:34ch; margin-bottom:26px;}
  .footer{display:flex; align-items:center; gap:18px;}
  .ghpill{display:inline-flex; align-items:center; gap:10px; font-family:var(--mono); font-size:15px;
    color:var(--ink); padding:9px 16px; border-radius:999px; border:1px solid var(--line-2);
    background:rgba(157,140,236,0.07);}
  .ghpill svg{width:18px;height:18px;fill:var(--primary);}
  .right{position:relative; overflow:hidden;}
  .right .device{position:absolute; top:-70px; left:44px; width:560px; transform:rotate(3deg);}
  .right .device img{width:100%;}
  .fade{position:absolute; inset:0; z-index:2; pointer-events:none;
    background:linear-gradient(90deg, var(--paper) 0%, transparent 22%),
              linear-gradient(0deg, rgba(7,4,23,0.5), transparent 42%);}
  `;
  const body = `<div class="wrap">
    <div class="left">
      <div>
        <div class="lockup">${WORDMARK("gh")}</div>
        <h1><span class="grad">Search, ask, and reason</span> over your own ChatGPT history.</h1>
        <div class="tag">A 100% local-first Chrome side panel + Python engine. Hybrid retrieval, cited answers, explainable "Ranked because" scoring, structured memory.</div>
        <div class="chips">
          <span class="chip">Loopback-only</span>
          <span class="chip">Hybrid search</span>
          <span class="chip">Grounded + cited</span>
          <span class="chip hot">Explainable ranking</span>
        </div>
      </div>
      <div class="footer">
        <span class="ghpill"><svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>github.com/SidharthAnand04/convsearch</span>
      </div>
    </div>
    <div class="right">
      <div class="device"><div class="screen"><img src="${PANEL_URI}"></div><div class="glowline"></div></div>
      <div class="fade"></div>
    </div>
  </div>`;
  return doc(1280, 640, body, css);
}

/* ---------------------------------------------------------------------------
 * Product Hunt gallery — 1270x760 each
 * ------------------------------------------------------------------------- */
const PH_W = 1270, PH_H = 760;

// header lockup row shared by PH slides
const phHeader = (idx, label) => `<div class="phhead">
  <div class="lockup">${WORDMARK("ph" + idx)}</div>
  <div class="eyebrow"><span class="dot"></span>${label}</div>
</div>`;

const phHeadCss = `
  .phhead{display:flex; align-items:center; justify-content:space-between;}
  .phhead .lockup{--wm:30px;}
`;

function ph01() {
  const css = phHeadCss + `
  .wrap{position:absolute; inset:0; padding:56px 72px 60px; display:grid;
    grid-template-rows:auto 1fr; }
  .hero{display:grid; grid-template-columns:1fr 400px; align-items:center; gap:40px; margin-top:14px;}
  .hero h1{font-size:66px; line-height:1.0; margin-bottom:26px;}
  .hero .tag{font-size:22px; line-height:1.5; color:var(--muted); max-width:26ch; margin-bottom:34px;}
  .hero .device{width:390px; margin:0 auto; transform:rotate(2.5deg);}
  .hero .device img{width:100%;}
  .chips .chip{font-size:14px; padding:9px 16px;}
  `;
  const body = `<div class="wrap">
    ${phHeader("01", "Your conversations, finally searchable")}
    <div class="hero">
      <div>
        <h1>Every decision you<br><span class="grad">talked your way into</span> — found in seconds.</h1>
        <div class="tag">convsearch turns your ChatGPT history into a private, searchable knowledge base. Live-capture-first. Nothing leaves your machine.</div>
        <div class="chips">
          <span class="chip">Hybrid keyword + semantic</span>
          <span class="chip">Grounded, cited answers</span>
          <span class="chip hot">100% local</span>
        </div>
      </div>
      <div class="device"><div class="screen"><img src="${PANEL_URI}"></div><div class="glowline"></div></div>
    </div>
  </div>`;
  return doc(PH_W, PH_H, body, css);
}

function ph02() {
  // Zoom on the "Ranked because" card. panel.png is 440x780.
  // Crop region approx: x 78..402 (w 324), y 300..672 (h 372) → the result card
  // with title, match chips, RANKED BECAUSE reason and the score mini-chart.
  const cropW = 324, cropH = 372, cropX = 78, cropY = 300;
  const scale = 640 / cropH; // fit crop height into ~640px frame
  const css = phHeadCss + `
  .wrap{position:absolute; inset:0; padding:56px 72px 56px; display:grid; grid-template-rows:auto 1fr;}
  .body{display:grid; grid-template-columns:520px 1fr; align-items:center; gap:60px; margin-top:6px;}
  .crop{width:${Math.round(cropW*scale)}px; height:${Math.round(cropH*scale)}px; border-radius:16px;
    overflow:hidden; position:relative; border:1px solid var(--line-2);
    box-shadow:0 40px 90px -30px rgba(0,0,0,0.85);}
  .crop .inner{width:${Math.round(440*scale)}px; height:${Math.round(780*scale)}px;
    background:url("${PANEL_URI}") no-repeat; background-size:100% 100%;
    transform:translate(${-Math.round(cropX*scale)}px, ${-Math.round(cropY*scale)}px);}
  .crop .sheen{position:absolute; inset:0; box-shadow:0 0 70px -12px rgba(222,58,144,0.45) inset; border-radius:inherit;}
  .copy h1{font-size:52px; line-height:1.03; margin-bottom:22px;}
  .copy .tag{font-size:21px; line-height:1.55; color:var(--muted); max-width:30ch; margin-bottom:30px;}
  .points{display:flex; flex-direction:column; gap:16px;}
  .pt{display:flex; gap:14px; align-items:flex-start;}
  .pt .k{font-family:var(--mono); font-size:12px; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--accent); border:1px solid rgba(222,58,144,0.4); border-radius:6px; padding:4px 8px; white-space:nowrap; margin-top:2px;}
  .pt .v{font-size:18px; line-height:1.45; color:var(--ink);}
  .pt .v b{color:#fff;}
  `;
  const body = `<div class="wrap">
    ${phHeader("02", "Explainable ranking")}
    <div class="body">
      <div class="crop"><div class="inner"></div><div class="sheen"></div></div>
      <div class="copy">
        <h1>Never wonder <span class="grad">why a result ranked</span>.</h1>
        <div class="tag">Every hit shows its reasoning in plain language — plus a live score breakdown you can open.</div>
        <div class="points">
          <div class="pt"><span class="k">Reason</span><span class="v"><b>"Ranked because"</b> — a sentence that names the exact keyword, semantic and title matches.</span></div>
          <div class="pt"><span class="k">Chart</span><span class="v">Per-signal <b>score breakdown</b>: keyword, meaning, title, overall.</span></div>
          <div class="pt"><span class="k">Honest</span><span class="v">Negative signals shown too — it never fakes confidence.</span></div>
        </div>
      </div>
    </div>
  </div>`;
  return doc(PH_W, PH_H, body, css);
}

function ph03() {
  const css = phHeadCss + `
  .wrap{position:absolute; inset:0; padding:56px 72px 60px; display:grid; grid-template-rows:auto auto 1fr;}
  .lead{margin-top:20px;}
  .lead h1{font-size:56px; line-height:1.02; margin-bottom:18px;}
  .lead .tag{font-size:21px; color:var(--muted); max-width:52ch; line-height:1.5;}
  .flow{display:grid; grid-template-columns:1fr auto 1fr auto 1fr; align-items:center; gap:0; margin-top:44px;}
  .node{border:1px solid var(--line-2); border-radius:18px; padding:30px 26px; background:rgba(20,16,49,0.72);
    box-shadow:0 24px 60px -30px rgba(0,0,0,0.8); min-height:210px; display:flex; flex-direction:column; gap:12px;}
  .node .ic{width:46px;height:46px;color:var(--primary);}
  .node h3{font-size:22px; letter-spacing:-0.01em;}
  .node p{font-size:15.5px; color:var(--muted); line-height:1.45;}
  .node .port{font-family:var(--mono); font-size:13px; color:var(--accent-2); letter-spacing:0.04em;}
  .arrow{display:flex; flex-direction:column; align-items:center; gap:8px; padding:0 20px;}
  .arrow .line{width:64px; height:2px; background:linear-gradient(90deg,var(--primary),var(--accent));}
  .arrow .lbl{font-family:var(--mono); font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:var(--faint);}
  .seal{margin-top:38px; display:inline-flex; align-items:center; gap:14px; align-self:start;
    padding:14px 22px; border-radius:999px; border:1px solid rgba(89,212,153,0.4); background:rgba(89,212,153,0.08);}
  .seal .d{width:9px;height:9px;border-radius:50%;background:var(--dot-online);box-shadow:0 0 12px var(--dot-online);}
  .seal span{font-size:17px; color:var(--ink);} .seal b{color:#8ff0bf;}
  `;
  // simple line icons
  const icChrome = `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 8h18"/><circle cx="6" cy="6" r="0.6" fill="currentColor"/><path d="M8 12h8M8 15h5"/></svg>`;
  const icEngine = `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><path d="M7 7h.01M7 17h.01"/></svg>`;
  const icDisk = `<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.66 3.58 3 8 3s8-1.34 8-3V6"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>`;
  const body = `<div class="wrap">
    ${phHeader("03", "Local-first architecture")}
    <div class="lead">
      <h1><span class="grad">Nothing leaves</span> your machine.</h1>
      <div class="tag">The panel talks to a Python engine over loopback only. Your history is indexed on your own disk. No account, no cloud — unless you explicitly opt into a cloud answer backend.</div>
    </div>
    <div>
      <div class="flow">
        <div class="node"><div>${icChrome}</div><h3>Chrome side panel</h3><p>Search, ask & reason as you browse. Live-capture-first, export optional.</p></div>
        <div class="arrow"><span class="lbl">127.0.0.1</span><span class="line"></span><span class="lbl">loopback</span></div>
        <div class="node"><div>${icEngine}</div><h3>Local Python engine</h3><p>Hybrid keyword + semantic index, grounded answers, ranking.</p><span class="port">bound to localhost</span></div>
        <div class="arrow"><span class="lbl">reads</span><span class="line"></span><span class="lbl">writes</span></div>
        <div class="node"><div>${icDisk}</div><h3>Your disk</h3><p>Conversations, embeddings & structured memory — stored on your machine.</p></div>
      </div>
      <div class="seal"><span class="d"></span><span><b>Loopback-only.</b> No telemetry. Apache-2.0, auditable.</span></div>
    </div>
  </div>`;
  return doc(PH_W, PH_H, body, css);
}

function ph04() {
  // Side-panel tour: full panel device on the left, labeled icon-rail callouts on the right.
  const rails = [
    ["Home", "at a glance"],
    ["Ask", "grounded, cited answers"],
    ["Plan", "turn threads into plans"],
    ["Tasks", "extracted to-dos"],
    ["Projects", "reconstructed context"],
    ["Timeline", "history by date"],
    ["Memories", "never-overwrite facts"],
    ["Review", "verify & curate"],
    ["Captures", "live-capture feed"],
    ["Privacy", "local-only controls"],
    ["Status", "engine health"],
  ];
  const css = phHeadCss + `
  .wrap{position:absolute; inset:0; padding:56px 72px 56px; display:grid; grid-template-rows:auto 1fr;}
  .body{display:grid; grid-template-columns:430px 1fr; align-items:center; gap:64px; margin-top:6px;}
  .stageL{display:flex; justify-content:center;}
  .device{width:400px; transform:rotate(-2deg);} .device img{width:100%;}
  .copy h1{font-size:46px; line-height:1.05; margin-bottom:12px;}
  .copy .tag{font-size:18px; color:var(--muted); line-height:1.5; max-width:40ch; margin-bottom:26px;}
  .rail{display:grid; grid-template-columns:1fr 1fr; gap:12px 22px;}
  .r{display:flex; gap:12px; align-items:baseline;}
  .r .n{font-family:var(--mono); font-size:12px; color:var(--faint);}
  .r .name{font-size:17px; color:var(--ink); font-weight:600; min-width:88px;}
  .r .desc{font-size:14.5px; color:var(--muted);}
  `;
  const railRows = rails.map(([name, desc], i) =>
    `<div class="r"><span class="n">${String(i + 1).padStart(2, "0")}</span><span class="name">${name}</span><span class="desc">${desc}</span></div>`
  ).join("");
  const body = `<div class="wrap">
    ${phHeader("04", "One panel, the whole workflow")}
    <div class="body">
      <div class="stageL"><div class="device"><div class="screen"><img src="${PANEL_URI}"></div><div class="glowline"></div></div></div>
      <div class="copy">
        <h1>A left icon rail for <span class="grad">everything you do</span>.</h1>
        <div class="tag">Eleven focused views, light or dark. From asking a question to reconstructing a whole project — without leaving the side panel.</div>
        <div class="rail">${railRows}</div>
      </div>
    </div>
  </div>`;
  return doc(PH_W, PH_H, body, css);
}

/* ---- render loop ---------------------------------------------------------- */

const JOBS = [
  { name: "og-card",         w: 1200, h: 630, out: path.join(SITE_MEDIA, "og-card.png"),            html: ogCard() },
  { name: "github-banner",   w: 1280, h: 640, out: path.join(LAUNCH_ASSETS, "github-banner.png"),   html: githubBanner() },
  { name: "ph-gallery-01",   w: PH_W, h: PH_H, out: path.join(LAUNCH_ASSETS, "ph-gallery-01.png"),  html: ph01() },
  { name: "ph-gallery-02",   w: PH_W, h: PH_H, out: path.join(LAUNCH_ASSETS, "ph-gallery-02.png"),  html: ph02() },
  { name: "ph-gallery-03",   w: PH_W, h: PH_H, out: path.join(LAUNCH_ASSETS, "ph-gallery-03.png"),  html: ph03() },
  { name: "ph-gallery-04",   w: PH_W, h: PH_H, out: path.join(LAUNCH_ASSETS, "ph-gallery-04.png"),  html: ph04() },
];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return `${b.readUInt32BE(16)}x${b.readUInt32BE(20)}`;
}

const browser = await chromium.launch();
try {
  for (const job of JOBS) {
    const page = await browser.newPage({ viewport: { width: job.w, height: job.h }, deviceScaleFactor: 1 });
    await page.setContent(job.html, { waitUntil: "networkidle" });
    await page.evaluate(() => document.fonts && document.fonts.ready);
    await page.screenshot({ path: job.out, clip: { x: 0, y: 0, width: job.w, height: job.h } });
    await page.close();
    console.log(`${job.name}  ->  ${path.relative(REPO, job.out).replace(/\\/g, "/")}  (${pngSize(job.out)})`);
  }
} finally {
  await browser.close();
}
