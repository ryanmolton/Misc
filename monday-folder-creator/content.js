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

  const boardId = (location.pathname.match(/boards\/(\d+)/) || [])[1];
  // Matches things like data-pulse-id="123…", id="row-pulse-123…", href=".../pulses/123…".
  const ITEM_ID_RE = /(?:pulse|item)[\w-]*?[=\/_-]\s*"?(\d{6,})/i;

  function idInAttributes(el) {
    for (const attr of el.attributes) {
      const m = `${attr.name}=${attr.value}`.match(ITEM_ID_RE);
      if (m && m[1] !== boardId) return m[1];
    }
    return null;
  }

  // Monday often tags a row's elements with the item's ID even when the
  // Item ID column isn't on screen. Look in the row around `el`.
  function itemIdFromRow(el) {
    for (let a = el, depth = 0; a && a !== document.body && depth < 15; a = a.parentElement, depth++) {
      if (a.getBoundingClientRect().height > 80) break; // past the row, into the group
      const found = idInAttributes(a) || [...a.querySelectorAll('a[href*="pulse"]')].map(idInAttributes).find(Boolean);
      if (found) return found;
    }
    return null;
  }

  // Values seen while the key column was on screen, so they can still be
  // used after it's scrolled away.
  const memoryKey = (keyColumn) => `mfc:${location.pathname}:${keyColumn}`;
  function loadMemory(keyColumn) {
    try {
      return JSON.parse(sessionStorage.getItem(memoryKey(keyColumn))) || {};
    } catch {
      return {};
    }
  }
  function saveMemory(keyColumn, memory) {
    try {
      sessionStorage.setItem(memoryKey(keyColumn), JSON.stringify(memory));
    } catch {
      // Storage full or blocked; remembering is only a convenience.
    }
  }
  const rowSignature = (values, keyColumn) =>
    JSON.stringify(Object.entries(values).filter(([c]) => c !== keyColumn).sort());

  // --- Reading Monday's own row markup (preferred) ---
  // Each item row is <div id="row-pulse-currentBoard-<board>-<item>-…">, and
  // each cell has a screen-reader label "<Column> <Item name> <Value>". This
  // works even for columns scrolled off screen, as long as the row is drawn.
  const isItemIdColumn = (c) => /^item\s*id$/i.test(norm(c));
  const NAME_ALIASES = ['name', 'item', 'item name'];

  function nameColumnTitle() {
    const el = document.getElementById('column-title-name');
    return el ? visibleText(el, hiddenChecker()) || norm(el.textContent) : null;
  }

  function rowElements() {
    return [...document.querySelectorAll('[id^="row-pulse-"]')].filter((row) => {
      if (/-placeholder$/.test(row.id) && !/-notplaceholder$/.test(row.id)) return false;
      return !boardId || row.id.includes(`-${boardId}-`); // skip subitem rows
    });
  }

  const itemIdOf = (row) => (row.id.match(/\d{6,}/g) || []).pop() || null;

  function itemNameOf(row, id) {
    const el = row.querySelector('[role="heading"]') || document.getElementById(`name-cell-${id}`);
    return el ? norm(el.textContent) : '';
  }

  function visibleText(root, isHidden) {
    const parts = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = norm(n.nodeValue);
      if (text && n.parentElement && !isHidden(n.parentElement)) parts.push(text);
    }
    return norm(parts.join(' '));
  }

  function cellValue(row, column, name, isHidden) {
    const prefix = `${lower(column)} ${lower(name)}`;
    for (const label of row.querySelectorAll('[class*="srOnly"]')) {
      const text = norm(label.textContent);
      const l = text.toLowerCase();
      if (l !== prefix && !l.startsWith(prefix + ' ')) continue;
      // Prefer what's shown in the cell (the label can carry extras like a colour name).
      const shown = label.parentElement ? visibleText(label.parentElement, isHidden) : '';
      return shown || text.slice(prefix.length).trim();
    }
    return null;
  }

  function scrapeStructured({ columns }) {
    const rowEls = rowElements();
    if (!rowEls.length) return null;
    const nameTitle = nameColumnTitle();
    const isHidden = hiddenChecker();
    const rows = [];
    const found = new Set();
    const seen = new Set();
    for (const row of rowEls) {
      const id = itemIdOf(row);
      const name = id && itemNameOf(row, id);
      if (!id || !name || seen.has(id)) continue;
      seen.add(id);
      const values = {};
      for (const c of columns) {
        let v = null;
        if (isItemIdColumn(c)) v = id;
        else if ((nameTitle && lower(c) === lower(nameTitle)) || NAME_ALIASES.includes(lower(c))) v = name;
        else v = cellValue(row, c, name, isHidden);
        if (v != null) found.add(c);
        values[c] = v || '';
      }
      rows.push(values);
    }
    return { rows, missingColumns: columns.filter((c) => !found.has(c)), method: 'rows' };
  }

  // --- Reading by position on screen (fallback) ---
  function scrape(msg) {
    const structured = scrapeStructured(msg);
    return structured && structured.rows.length ? structured : scrapeByPosition(msg);
  }

  function scrapeByPosition({ keyColumn, columns }) {
    const headers = {};
    for (const c of columns) headers[c] = headerCells(c);
    // Rows are found from the key column, or any other visible column if it's off screen.
    const anchor = headers[keyColumn] && headers[keyColumn].length ? keyColumn : columns.find((c) => headers[c].length);
    if (!anchor) return { rows: [], missingColumns: columns };

    // Header titles themselves (incl. other groups' headers) aren't cells.
    const allHeaders = Object.values(headers).flat();
    const inHeader = (it) =>
      allHeaders.some((h) => it.cx >= h.left && it.cx <= h.right && it.cy >= h.top && it.cy <= h.bottom);
    const items = collectText().filter((it) => !inHeader(it));

    // Rows: group the anchor column's text by height on the page.
    const anchorRows = [];
    for (const it of items) {
      const h = headerFor(headers[anchor], it);
      if (!h) continue;
      const row = anchorRows.find((r) => r.header === h && Math.abs(r.cy - it.cy) < 6);
      if (row) row.parts.push(it.text);
      else anchorRows.push({ header: h, cy: it.cy, el: it.el, parts: [it.text] });
    }

    const memory = loadMemory(keyColumn);
    const rows = [];
    const seen = new Set();
    for (const ar of anchorRows) {
      const anchorValue = joinParts(ar.parts);
      if (!anchorValue || /^\+\s*add\b/i.test(anchorValue)) continue; // the "+ Add project" row
      const values = { [anchor]: anchorValue };
      for (const c of columns) {
        if (c === anchor) continue;
        const parts = items
          .filter((it) => Math.abs(it.cy - ar.cy) <= ROW_TOLERANCE && headerFor(headers[c], it))
          .map((it) => it.text);
        values[c] = joinParts(parts);
      }
      // A row with only one value is usually the group's summary footer.
      const others = columns.filter((c) => c !== anchor && headers[c].length);
      if (others.length && others.every((c) => !values[c])) continue;

      const sig = rowSignature(values, keyColumn);
      if (values[keyColumn]) {
        memory[sig] = values[keyColumn];
      } else {
        values[keyColumn] = memory[sig] || (/\bid\b/i.test(keyColumn) && itemIdFromRow(ar.el)) || '';
      }
      const id = values[keyColumn] || sig;
      if (seen.has(id)) continue;
      seen.add(id);
      rows.push(values);
    }
    saveMemory(keyColumn, memory);

    const missingColumns = columns.filter((c) => !headers[c].length && !(rows.length && rows.every((r) => r[c])));
    return { rows, missingColumns, method: 'position' };
  }

  // Everything the extension sees around the first rows, for troubleshooting.
  function debugInfo(msg) {
    const round = (r) => [r.left, r.top, r.width, r.height].map(Math.round);
    const headers = {};
    for (const c of msg.columns) headers[c] = headerCells(c).map(round);
    const anchor = msg.columns.find((c) => headers[c].length);
    const first = anchor && headerCells(anchor)[0];
    const near = [];
    let firstRowEl = null;
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
        if (!firstRowEl && r.top > first.bottom && r.left >= first.left && r.left <= first.right) firstRowEl = el;
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
    // The first row's surrounding elements and their attributes.
    const ancestors = [];
    for (let a = firstRowEl, d = 0; a && a !== document.body && d < 18; a = a.parentElement, d++) {
      const attrs = {};
      for (const at of a.attributes) attrs[at.name] = at.value.slice(0, 120);
      ancestors.push({ tag: a.tagName.toLowerCase(), h: Math.round(a.getBoundingClientRect().height), attrs });
    }
    const links = firstRowEl
      ? [...(firstRowEl.closest('[role="row"]') || firstRowEl.parentElement).querySelectorAll('a[href]')].map((a) => a.getAttribute('href').slice(0, 120)).slice(0, 10)
      : [];
    const firstRow = rowElements()[0];
    const rowInfo = {
      rowCount: rowElements().length,
      nameTitle: nameColumnTitle(),
      firstRowId: firstRow ? firstRow.id : null,
      firstRowLabels: firstRow ? [...firstRow.querySelectorAll('[class*="srOnly"]')].map((l) => norm(l.textContent).slice(0, 100)) : [],
    };
    return { page: location.pathname, rowInfo, headers, near: near.slice(0, 60), ancestors, links, result: scrape(msg) };
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
