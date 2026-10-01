// Helpers shared by the side panel and the settings page.
const MFC = (() => {
  const DEFAULTS = {
    pattern: '{Item ID}-{Project}',
    matchColumn: 'Item ID',
    rootPath: '',
    renameInTemplate: true,
  };

  async function getSettings() {
    return { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
  }

  async function saveSettings(settings) {
    await chrome.storage.sync.set(settings);
  }

  // Folder handles can't go in chrome.storage, but IndexedDB can hold them.
  function openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('monday-folder-creator', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('handles');
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function getHandle(key) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const req = db.transaction('handles').objectStore('handles').get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  async function setHandle(key, handle) {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('handles', 'readwrite');
      tx.objectStore('handles').put(handle, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function hasPermission(handle, mode) {
    return (await handle.queryPermission({ mode })) === 'granted';
  }

  // Must be called from a click handler: Chrome shows a prompt if needed.
  async function requestPermission(handle, mode) {
    if (await hasPermission(handle, mode)) return true;
    return (await handle.requestPermission({ mode })) === 'granted';
  }

  function tokens(pattern) {
    return [...new Set([...pattern.matchAll(/\{([^{}]+)\}/g)].map((m) => m[1].trim()))];
  }

  // The column used to spot rows on the page and to find existing folders.
  function keyColumn(settings) {
    return settings.matchColumn.trim() || tokens(settings.pattern).find((t) => !isBuiltin(t));
  }

  function columnsNeeded(settings) {
    const cols = tokens(settings.pattern).filter((t) => !isBuiltin(t));
    const key = keyColumn(settings);
    if (key && !cols.some((c) => c.toLowerCase() === key.toLowerCase())) cols.push(key);
    return cols;
  }

  function isBuiltin(name) {
    return name.toLowerCase() === 'today';
  }

  function today() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }

  function lookup(values, name) {
    const key = Object.keys(values).find((k) => k.toLowerCase() === name.toLowerCase());
    if (key !== undefined) return values[key];
    if (isBuiltin(name)) return today();
    return null;
  }

  // Strip characters Windows or macOS won't accept in a file or folder name.
  function sanitize(name) {
    let out = String(name)
      .replace(/"/g, "'")
      .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '-')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 150)
      .replace(/[. ]+$/, '');
    if (/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(out)) out += '_';
    return out;
  }

  // A "/" in the pattern makes nested folders, e.g. "{Group}/{Item ID}-{Project}".
  function buildSegments(pattern, values) {
    const missing = [];
    const segments = pattern
      .split(/[\\/]/)
      .map((seg) =>
        sanitize(
          seg.replace(/\{([^{}]+)\}/g, (_, raw) => {
            const name = raw.trim();
            const value = lookup(values, name);
            if (value == null || value === '') {
              missing.push(name);
              return '';
            }
            return value.replace(/[\\/]/g, '-');
          })
        )
      )
      .filter(Boolean);
    return { segments, missing };
  }

  async function getDirIfExists(root, segments) {
    let dir = root;
    for (const name of segments) {
      try {
        dir = await dir.getDirectoryHandle(name);
      } catch (e) {
        if (e.name === 'NotFoundError' || e.name === 'TypeMismatchError') return null;
        throw e;
      }
    }
    return dir;
  }

  async function listDirs(dir) {
    const names = [];
    for await (const [name, entry] of dir.entries()) {
      if (entry.kind === 'directory') names.push(name);
    }
    return names;
  }

  // Exact name first; otherwise any folder that starts with the key value,
  // so a project folder renamed after creation is still found.
  function findExisting(names, expected, keyValue) {
    const exp = expected.toLowerCase();
    const exact = names.find((n) => n.toLowerCase() === exp);
    if (exact) return exact;
    if (!keyValue) return null;
    const key = sanitize(keyValue).toLowerCase();
    if (!key) return null;
    return (
      names.find((n) => {
        const l = n.toLowerCase();
        return l === key || (l.startsWith(key) && !/[a-z0-9]/i.test(l[key.length]));
      }) || null
    );
  }

  const SKIP_FILES = new Set(['.ds_store', 'thumbs.db', 'desktop.ini']);

  async function copyDir(src, dst, values, renameInTemplate) {
    for await (const [name, entry] of src.entries()) {
      if (SKIP_FILES.has(name.toLowerCase()) || name.startsWith('~$')) continue;
      let target = name;
      if (renameInTemplate && name.includes('{')) {
        const filled = name.replace(/\{([^{}]+)\}/g, (m, raw) => {
          const value = lookup(values, raw.trim());
          return value == null ? m : value.replace(/[\\/]/g, '-');
        });
        target = sanitize(filled) || name;
      }
      if (entry.kind === 'directory') {
        const sub = await dst.getDirectoryHandle(target, { create: true });
        await copyDir(entry, sub, values, renameInTemplate);
      } else {
        const file = await entry.getFile();
        const out = await (await dst.getFileHandle(target, { create: true })).createWritable();
        await out.write(file);
        await out.close();
      }
    }
  }

  // Returns a problem description, or null if the two folders can be used together.
  async function checkFolderPair(root, template) {
    if (await root.isSameEntry(template)) return 'The template and project folders are the same folder.';
    if (await template.resolve(root)) return 'The project folder is inside the template folder, so copying it would never end.';
    return null;
  }

  async function createProject({ root, template, segments, values, renameInTemplate }) {
    const problem = await checkFolderPair(root, template);
    if (problem) throw new Error(problem);
    let parent = root;
    for (const name of segments.slice(0, -1)) {
      parent = await parent.getDirectoryHandle(name, { create: true });
    }
    const name = segments[segments.length - 1];
    if (await getDirIfExists(parent, [name])) throw new Error(`"${name}" already exists.`);
    const dst = await parent.getDirectoryHandle(name, { create: true });
    try {
      await copyDir(template, dst, values, renameInTemplate);
    } catch (e) {
      // Don't leave a half-copied project folder behind.
      await parent.removeEntry(name, { recursive: true }).catch(() => {});
      throw e;
    }
  }

  return {
    DEFAULTS,
    getSettings,
    saveSettings,
    getHandle,
    setHandle,
    hasPermission,
    requestPermission,
    tokens,
    keyColumn,
    columnsNeeded,
    sanitize,
    buildSegments,
    getDirIfExists,
    listDirs,
    findExisting,
    copyDir,
    checkFolderPair,
    createProject,
  };
})();
