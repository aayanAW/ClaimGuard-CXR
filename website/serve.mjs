// serve.mjs — minimal static HTTP server for the ClaimGuard-CXR website.
// Serves the project root on http://localhost:3000. No dependencies beyond Node built-ins.
//
// Usage:
//   node serve.mjs           # default port 3000
//   PORT=4000 node serve.mjs # custom port
//
// Starts in foreground; run with & or nohup for background.

import http from "node:http";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
  ".otf": "font/otf",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

function sanitize(urlPath) {
  // Strip query string and hash, decode, and resolve inside ROOT only.
  const clean = decodeURIComponent(urlPath.split("?")[0].split("#")[0]);
  const resolved = path.normalize(path.join(ROOT, clean));
  if (!resolved.startsWith(ROOT)) return null; // directory traversal attempt
  return resolved;
}

const server = http.createServer(async (req, res) => {
  try {
    let filePath = sanitize(req.url || "/");
    if (!filePath) {
      res.writeHead(403, { "Content-Type": "text/plain" });
      res.end("Forbidden");
      return;
    }

    // If path is a directory, serve index.html inside it.
    let stat;
    try {
      stat = await fs.stat(filePath);
    } catch {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not Found");
      return;
    }

    if (stat.isDirectory()) {
      filePath = path.join(filePath, "index.html");
      try {
        stat = await fs.stat(filePath);
      } catch {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not Found");
        return;
      }
    }

    const ext = path.extname(filePath).toLowerCase();
    const mime = MIME[ext] || "application/octet-stream";
    const data = await fs.readFile(filePath);

    res.writeHead(200, {
      "Content-Type": mime,
      "Content-Length": data.length,
      // Disable caching during development so reloads always show the latest.
      "Cache-Control": "no-store",
    });
    res.end(data);
  } catch (err) {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end(`Internal Server Error: ${err.message}`);
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Listening on http://localhost:${PORT}`);
  console.log(`Root: ${ROOT}`);
});

// Graceful shutdown on SIGTERM / SIGINT so background runs can be killed cleanly.
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    server.close(() => {
      console.log(`\nServer stopped (${sig}).`);
      process.exit(0);
    });
  });
}
