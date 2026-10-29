#!/usr/bin/env node
// Recomputes the CSP hashes in vsdx-viewer.html for its inline <script> and <style>.
// Usage: node build-csp.js [path/to/vsdx-viewer.html]
"use strict";
const fs = require("fs");
const crypto = require("crypto");

const file = process.argv[2] || require("path").join(__dirname, "vsdx-viewer.html");
let html = fs.readFileSync(file, "utf8");
const hash = body => "sha256-" + crypto.createHash("sha256").update(body, "utf8").digest("base64");
const body = tag => {
  const m = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`).exec(html);
  if (!m) throw new Error(`no inline <${tag}> found`);
  return m[1];
};
const scriptHash = hash(body("script")), styleHash = hash(body("style"));
html = html
  .replace(/script-src '[^']*'/, `script-src '${scriptHash}'`)
  .replace(/style-src '[^']*'/, `style-src '${styleHash}'`);
fs.writeFileSync(file, html);
console.log(`script-src '${scriptHash}'\nstyle-src  '${styleHash}'`);
