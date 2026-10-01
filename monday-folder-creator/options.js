const $ = (s) => document.querySelector(s);
const SAMPLE = { 'Item ID': '12111781128', Project: 'Enterprise-IsatPhone 3-Video-Asset-2026' };

async function init() {
  const s = await MFC.getSettings();
  $('#pattern').value = s.pattern;
  $('#matchColumn').value = s.matchColumn;
  $('#rootPath').value = s.rootPath;
  $('#renameInTemplate').checked = s.renameInTemplate;
  $('#pattern').oninput = preview;
  preview();

  showHandle('template', await MFC.getHandle('template'));
  showHandle('root', await MFC.getHandle('root'));
  $('#pickTemplate').onclick = () => pick('template', 'read');
  $('#pickRoot').onclick = () => pick('root', 'readwrite');
  $('#save').onclick = save;
}

function preview() {
  const values = {};
  for (const t of MFC.tokens($('#pattern').value)) values[t] = SAMPLE[t] ?? `<${t}>`;
  $('#preview').textContent = MFC.buildSegments($('#pattern').value, values).segments.join(' / ') || '—';
}

function showHandle(key, handle) {
  const el = $(key === 'template' ? '#templateName' : '#rootName');
  el.textContent = handle ? handle.name : 'Not chosen';
  el.className = handle ? '' : 'muted';
}

async function pick(key, mode) {
  let handle;
  try {
    handle = await showDirectoryPicker({ id: `mfc-${key}`, mode });
  } catch (e) {
    if (e.name !== 'AbortError') alert(e.message);
    return;
  }
  const other = await MFC.getHandle(key === 'template' ? 'root' : 'template');
  if (other) {
    const problem = await MFC.checkFolderPair(key === 'root' ? handle : other, key === 'root' ? other : handle);
    if (problem) {
      alert(problem);
      return;
    }
  }
  await MFC.setHandle(key, handle);
  showHandle(key, handle);
  // Lets an open side panel know the folders changed.
  await chrome.storage.local.set({ foldersChangedAt: Date.now() });
}

async function save() {
  const pattern = $('#pattern').value.trim();
  if (!MFC.tokens(pattern).length) {
    alert('The folder name needs at least one {column}.');
    return;
  }
  await MFC.saveSettings({
    pattern,
    matchColumn: $('#matchColumn').value.trim(),
    rootPath: $('#rootPath').value.trim(),
    renameInTemplate: $('#renameInTemplate').checked,
  });
  $('#saved').textContent = 'Saved.';
  setTimeout(() => ($('#saved').textContent = ''), 2000);
}

init();
