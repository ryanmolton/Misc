const $ = (s) => document.querySelector(s);
const CACHE_MS = 5000;

let settings;
let rootHandle = null;
let templateHandle = null;
let tabId = null;
let rows = [];
let missingColumns = [];
const dirCache = new Map();
const busy = new Set();

async function loadConfig() {
  settings = await MFC.getSettings();
  [rootHandle, templateHandle] = await Promise.all([MFC.getHandle('root'), MFC.getHandle('template')]);
  dirCache.clear();
}

async function init() {
  await loadConfig();
  $('#refresh').onclick = () => {
    dirCache.clear();
    refresh();
  };
  $('#settings').onclick = () => chrome.runtime.openOptionsPage();
  $('#filter').oninput = render;
  $('#debug').onclick = copyDebug;
  chrome.tabs.onActivated.addListener(scheduleRefresh);
  chrome.tabs.onUpdated.addListener((id, info) => {
    if (id === tabId && (info.status === 'complete' || info.url)) scheduleRefresh();
  });
  chrome.runtime.onMessage.addListener((msg, sender) => {
    if (msg.type === 'page-changed' && sender.tab && sender.tab.id === tabId && !document.hidden) scheduleRefresh();
  });
  // Settings page saved something (folders are signalled via storage.local).
  chrome.storage.onChanged.addListener(async () => {
    await loadConfig();
    refresh();
  });
  document.addEventListener('visibilitychange', () => !document.hidden && refresh());
  refresh();
}

let refreshTimer = null;
let refreshing = false;
let refreshAgain = false;

function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refresh, 300);
}

async function refresh() {
  if (refreshing) {
    refreshAgain = true;
    return;
  }
  refreshing = true;
  try {
    await doRefresh();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    refreshing = false;
    if (refreshAgain) {
      refreshAgain = false;
      scheduleRefresh();
    }
  }
}

async function doRefresh() {
  const ready = await updateBanner();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = tab ? tab.id : null;
  if (!tab || !/^https:\/\/([^/]+\.)?monday\.com\//.test(tab.url || '')) {
    rows = [];
    missingColumns = [];
    setStatus('Open a Monday.com board in this tab to see its projects.');
    render();
    return;
  }

  const key = MFC.keyColumn(settings);
  const result = await scrape(tab.id, { keyColumn: key, columns: MFC.columnsNeeded(settings) });
  missingColumns = result.missingColumns;
  const next = result.rows.map((values) => {
    const { segments, missing } = MFC.buildSegments(settings.pattern, values);
    return { values, segments, missing, key: values[key], found: null, checked: false };
  });

  if (ready) {
    for (const row of next) {
      if (row.missing.length || !row.segments.length) continue;
      const names = await listParent(row.segments.slice(0, -1));
      row.found = MFC.findExisting(names, row.segments[row.segments.length - 1], row.key);
      row.checked = true;
    }
  }
  rows = next;

  const lacking = rows.filter((r) => r.missing.length).length;
  if (lacking) {
    const cols = [...new Set(rows.flatMap((r) => r.missing))].join(', ');
    setStatus(`${cols} isn't on screen for ${lacking} project${lacking === 1 ? '' : 's'}. Scroll the board sideways until ${cols} shows (it's remembered after that), or drag that column next to Project.`, true);
  } else if (missingColumns.length) {
    setStatus(`Can't see these columns on the page: ${missingColumns.join(', ')}. Scroll so they're visible, or check the names in Settings.`, true);
  } else if (!rows.length) {
    setStatus('No projects found on screen.');
  } else {
    setStatus(`${rows.length} project${rows.length === 1 ? '' : 's'} on screen.`);
  }
  render();
}

async function copyDebug() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const opts = { keyColumn: MFC.keyColumn(settings), columns: MFC.columnsNeeded(settings) };
    const info = await scrape(tab.id, opts, 'debug');
    const report = { version: chrome.runtime.getManifest().version, settings, ...info };
    await navigator.clipboard.writeText(JSON.stringify(report, null, 1));
    toast('Debug info copied. Paste it to whoever is helping you.');
  } catch (e) {
    toast(`Couldn't collect debug info: ${e.message}`);
  }
}

async function scrape(id, opts, type = 'scrape') {
  const msg = { type, ...opts };
  try {
    return await chrome.tabs.sendMessage(id, msg);
  } catch {
    // Tab was open before the extension was installed/reloaded.
    await chrome.scripting.executeScript({ target: { tabId: id }, files: ['content.js'] });
    return await chrome.tabs.sendMessage(id, msg);
  }
}

async function listParent(segments) {
  const cacheKey = segments.join('/').toLowerCase();
  const hit = dirCache.get(cacheKey);
  if (hit && Date.now() - hit.time < CACHE_MS) return hit.names;
  const dir = await MFC.getDirIfExists(rootHandle, segments);
  const names = dir ? await MFC.listDirs(dir) : [];
  dirCache.set(cacheKey, { time: Date.now(), names });
  return names;
}

// Returns true when both folders are chosen and usable.
async function updateBanner() {
  const banner = $('#banner');
  banner.replaceChildren();
  if (!rootHandle || !templateHandle) {
    banner.append('Choose the template folder and project folders location to get started.');
    banner.append(button('Open settings', () => chrome.runtime.openOptionsPage(), true));
    banner.hidden = false;
    return false;
  }
  const ok =
    (await MFC.hasPermission(rootHandle, 'readwrite')) && (await MFC.hasPermission(templateHandle, 'read'));
  if (!ok) {
    banner.append('Chrome needs your OK to use the project folders again.');
    banner.append(button('Allow folder access', async () => {
      if (await grantAccess()) refresh();
    }, true));
    banner.hidden = false;
    return false;
  }
  banner.hidden = true;
  return true;
}

async function grantAccess() {
  try {
    return (
      (await MFC.requestPermission(rootHandle, 'readwrite')) &&
      (await MFC.requestPermission(templateHandle, 'read'))
    );
  } catch (e) {
    toast(`Couldn't get folder access: ${e.message}`);
    return false;
  }
}

function render() {
  const filter = $('#filter').value.trim().toLowerCase();
  const list = $('#rows');
  list.replaceChildren();
  for (const row of rows) {
    const label = row.segments.join(' / ');
    if (filter && !label.toLowerCase().includes(filter)) continue;
    const li = document.createElement('li');

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = label || '(no name)';
    li.append(name);

    const expected = row.segments[row.segments.length - 1];
    if (row.found && row.found !== expected) li.append(sub(`Found as: ${row.found}`));
    if (row.missing.length) li.append(sub(`Missing on page: ${row.missing.join(', ')}`, true));

    const foot = document.createElement('div');
    foot.className = 'row-foot';
    const chip = document.createElement('span');
    chip.className = 'chip' + (row.found ? ' ok' : '');
    chip.textContent = row.found ? 'Folder exists' : row.checked ? 'No folder yet' : '—';
    foot.append(chip);

    const actions = document.createElement('div');
    actions.className = 'actions';
    if (row.found) {
      actions.append(button('Copy path', () => copyPath(row)));
    } else if (row.checked) {
      const b = button(busy.has(row.key) ? 'Creating…' : 'Create', () => create(row), true);
      b.disabled = busy.has(row.key);
      actions.append(b);
    }
    foot.append(actions);
    li.append(foot);
    list.append(li);
  }
}

async function create(row) {
  if (busy.has(row.key)) return;
  if (!(await grantAccess())) return;
  busy.add(row.key);
  render();
  try {
    await MFC.createProject({
      root: rootHandle,
      template: templateHandle,
      segments: row.segments,
      values: row.values,
      renameInTemplate: settings.renameInTemplate,
    });
    toast(`Created "${row.segments[row.segments.length - 1]}"`);
  } catch (e) {
    toast(`Couldn't create folder: ${e.message}`);
  } finally {
    busy.delete(row.key);
    dirCache.clear();
    refresh();
  }
}

async function copyPath(row) {
  const base = settings.rootPath.trim();
  if (!base) {
    toast('Add the full path of the project folders location in Settings to copy paths.');
    return;
  }
  const sep = base.includes('\\') && !base.includes('/') ? '\\' : '/';
  const path = [base.replace(/[\\/]+$/, ''), ...row.segments.slice(0, -1), row.found].join(sep);
  try {
    await navigator.clipboard.writeText(path);
    toast('Path copied. Paste it into File Explorer / Finder (Go ▸ Go to Folder).');
  } catch (e) {
    toast(`Couldn't copy: ${e.message}`);
  }
}

function button(text, onClick, primary = false) {
  const b = document.createElement('button');
  b.textContent = text;
  if (primary) b.className = 'primary';
  b.onclick = onClick;
  return b;
}

function sub(text, isError = false) {
  const d = document.createElement('div');
  d.className = 'sub' + (isError ? ' error' : '');
  d.textContent = text;
  return d;
}

function setStatus(text, isError = false) {
  const s = $('#status');
  s.textContent = text;
  s.className = isError ? 'error' : 'muted';
}

let toastTimer = null;
function toast(text) {
  const t = $('#toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4000);
}

init();
