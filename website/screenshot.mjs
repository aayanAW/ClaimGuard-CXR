// screenshot.mjs — Puppeteer headless Chrome screenshot helper.
//
// Usage:
//   node screenshot.mjs <url> [label]
//
// Examples:
//   node screenshot.mjs http://localhost:3000
//   node screenshot.mjs http://localhost:3000 hero-v1
//   node screenshot.mjs http://localhost:3000 mobile  (use with WIDTH=375 HEIGHT=812)
//
// Environment variables:
//   WIDTH       viewport width (default 1440)
//   HEIGHT      viewport height (default 900)
//   FULL_PAGE   'true' (default) for full-page capture, 'false' for viewport-only
//   DPR         device pixel ratio (default 2 for retina screenshots)
//   WAIT        'networkidle0' (default) / 'networkidle2' / 'domcontentloaded' / 'load'
//   DELAY       extra ms to wait after navigation, before screenshotting (default 400)
//
// Output:
//   Saved to ./temporary screenshots/screenshot-N[-label].png
//   N auto-increments from the highest existing number in that directory.

import puppeteer from "puppeteer";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error("Usage: node screenshot.mjs <url> [label]");
    process.exit(1);
  }

  const url = args[0];
  const label = args[1] || "";

  const width = Number(process.env.WIDTH) || 1440;
  const height = Number(process.env.HEIGHT) || 900;
  const fullPage = (process.env.FULL_PAGE || "true") !== "false";
  const dpr = Number(process.env.DPR) || 2;
  const waitUntil = process.env.WAIT || "networkidle0";

  const outDir = path.join(__dirname, "temporary screenshots");
  await fs.mkdir(outDir, { recursive: true });

  // Find next sequential number
  const entries = await fs.readdir(outDir);
  let maxN = 0;
  const pattern = /^screenshot-(\d+)(?:-[^.]+)?\.png$/;
  for (const name of entries) {
    const m = name.match(pattern);
    if (m) {
      const n = parseInt(m[1], 10);
      if (n > maxN) maxN = n;
    }
  }
  const n = maxN + 1;
  const filename = label ? `screenshot-${n}-${label}.png` : `screenshot-${n}.png`;
  const outPath = path.join(outDir, filename);

  console.log(`Launching Chromium (${width}x${height}, DPR=${dpr}, fullPage=${fullPage})...`);
  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
    ],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({
      width,
      height,
      deviceScaleFactor: dpr,
    });

    console.log(`Navigating to ${url}...`);
    await page.goto(url, { waitUntil, timeout: 30000 });

    // Extra idle to let fonts + animations settle
    const delay = Number(process.env.DELAY) || 400;
    await new Promise((r) => setTimeout(r, delay));

    await page.screenshot({
      path: outPath,
      fullPage,
      type: "png",
    });

    console.log(`Saved: ${outPath}`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error("Screenshot failed:", err);
  process.exit(1);
});
