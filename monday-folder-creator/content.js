// Reads the Monday board table as it appears on screen. Monday's class names
// change often, so this works from what is visible instead: it finds the
// column headers by their text, then reads the cells lined up under them.
(() => {
  if (window.__mondayFolderCreator) return;
  window.__mondayFolderCreator = true;

  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const lower = (s) => norm(s).toLowerCase();
  const ROW_TOLERANCE = 14; // px between a row's key cell and its other cells

  // Screen-reader-only text (e.g. Monday's "Proof link <item> <url>" labels)
  // is in the page but not on screen, so it mustn't be read as a cell.
  function hiddenChecker() {
    const cache = new Map();
    const hidden = (el, depth = 0) => {
      if (!el || el === document.body || depth > 6) return false;
      if (cache.has(el)) return cache.get(el);
      const st = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      const result =
        st.visibility === 'hidden' ||
        st.opacity === '0' ||
        (st.clip !== 'auto' && st.position === 'absolute') ||
        /inset\((50|100)%/.test(st.clipPath) ||
        (st.overflow !== 'visible' && (r.width <= 2 || r.height <= 2)) ||
        hidden(el.parentElement, depth + 1);
      cache.set(el, result);
      return result;
    };
    return hidden;
  }

  function collectText() {
    const items = [];
    const isHidden = hiddenChecker();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = norm(n.nodeValue);
      if (!text) continue;
      const el = n.parentElement;
      if (!el || el.closest('script, style, noscript, template')) continue;
      range.selectNodeContents(n);
      const r = range.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      // Long names are cut off with "…"; use the visible part for position.
      const er = el.getBoundingClientRect();
      const left = Math.max(r.left, er.left);
      const right = Math.min(r.right, er.right);
      if (right - left < 2 || isHidden(el)) continue;
      items.push({ el, text, cx: (left + right) / 2, top: r.top, cy: (r.top + r.bottom) / 2 });
    }
    return items;
  }

  // Every header cell titled `name` (one per group on the board).
  function headerCells(name) {
    const want = lower(name);
    const cells = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (lower(n.nodeValue) !== want || !n.parentElement) continue;
      // Widen to the whole header cell: the biggest box showing only this title.
      let cell = n.parentElement;
      while (
        cell.parentElement &&
        cell.parentElement !== document.body &&
        lower(cell.parentElement.innerText) === want
      ) {
        cell = cell.parentElement;
      }
      const r = cell.getBoundingClientRect();
      if (r.width && r.height) cells.push(r);
    }
    return cells;
  }

  // The nearest header above the text whose column it sits in.
  function headerFor(headers, item) {
    let best = null;
    for (const h of headers) {
      if (item.cx < h.left || item.cx > h.right || h.bottom > item.top + 2) continue;
      if (!best || h.bottom > best.bottom) best = h;
    }
    return best;
  }

  function joinParts(parts) {
    // Drop little count badges (e.g. the "1" on the updates bubble).
    const words = parts.filter((p) => !/^\d{1,3}$/.test(p));
    return norm((words.length && words.length < parts.length ? words : parts).join(' '));
  }

  function scrape({ keyColumn, columns }) {
    const headers = {};
    for (const c of columns) headers[c] = headerCells(c);
    const missingColumns = columns.filter((c) => !headers[c].length);
    if (!headers[keyColumn] || !headers[keyColumn].length) return { rows: [], missingColumns };

    // Header titles themselves (incl. other groups' headers) aren't cells.
    const allHeaders = Object.values(headers).flat();
    const inHeader = (it) =>
      allHeaders.some((h) => it.cx >= h.left && it.cx <= h.right && it.cy >= h.top && it.cy <= h.bottom);
    const items = collectText().filter((it) => !inHeader(it));

    // Rows: group the key column's text by height on the page.
    const keyRows = [];
    for (const it of items) {
      const h = headerFor(headers[keyColumn], it);
      if (!h) continue;
      const row = keyRows.find((r) => r.header === h && Math.abs(r.cy - it.cy) < 6);
      if (row) row.parts.push(it.text);
      else keyRows.push({ header: h, cy: it.cy, parts: [it.text] });
    }

    const rows = [];
    const seen = new Set();
    for (const kr of keyRows) {
      const key = joinParts(kr.parts);
      if (!key || seen.has(key)) continue;
      const values = { [keyColumn]: key };
      for (const c of columns) {
        if (c === keyColumn) continue;
        const parts = items
          .filter((it) => Math.abs(it.cy - kr.cy) <= ROW_TOLERANCE && headerFor(headers[c], it))
          .map((it) => it.text);
        values[c] = joinParts(parts);
      }
      // A row with only a key value is usually the group's summary footer.
      if (columns.length > 1 && columns.every((c) => c === keyColumn || !values[c])) continue;
      seen.add(key);
      rows.push(values);
    }
    return { rows, missingColumns };
  }

  // Everything the extension sees around the first rows, for troubleshooting.
  function debugInfo(msg) {
    const round = (r) => [r.left, r.top, r.width, r.height].map(Math.round);
    const headers = {};
    for (const c of msg.columns) headers[c] = headerCells(c).map(round);
    const first = headerCells(msg.keyColumn)[0];
    const near = [];
    if (first) {
      const isHidden = hiddenChecker();
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const range = document.createRange();
      for (let n = walker.nextNode(); n && near.length < 200; n = walker.nextNode()) {
        const el = n.parentElement;
        if (!el || !norm(n.nodeValue) || el.closest('script, style, noscript, template')) continue;
        range.selectNodeContents(n);
        const r = range.getBoundingClientRect();
        if (r.bottom < first.top - 5 || r.top > first.bottom + 160) continue;
        near.push({
          text: norm(n.nodeValue).slice(0, 80),
          tag: el.tagName.toLowerCase(),
          cls: String(el.className).slice(0, 80),
          role: el.closest('[role]')?.getAttribute('role') || '',
          textBox: round(r),
          elBox: round(el.getBoundingClientRect()),
          hidden: isHidden(el),
        });
      }
    }
    return { page: location.pathname, headers, near, result: scrape(msg) };
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === 'scrape') sendResponse(scrape(msg));
    if (msg.type === 'debug') sendResponse(debugInfo(msg));
  });

  // Tell the side panel when the board changes or scrolls (at most ~1/sec).
  let timer = null;
  let last = 0;
  function notify() {
    if (timer) return;
    timer = setTimeout(() => {
      timer = null;
      last = Date.now();
      try {
        chrome.runtime.sendMessage({ type: 'page-changed' }).catch(() => {});
      } catch {
        // Extension was reloaded; this old copy of the script is orphaned.
      }
    }, Math.max(0, 1000 - (Date.now() - last)));
  }
  new MutationObserver(notify).observe(document.body, { childList: true, subtree: true, characterData: true });
  window.addEventListener('scroll', notify, { capture: true, passive: true });
})();
