/* Heliotrope stereoscope — prototype print kit.
 * Derives a fastenable, supportless-printable version of every part: dowel holes, heat-set insert
 * holes, counterbores, magnet pockets, grooves and nut traps are cut with a small BSP CSG, oversize
 * plates are split with seam dowels, and a few parts are split or reshaped so they print without
 * supports. Also returns every fastener (for the 3D view) and the hardware list.
 */
(function (root) {
  const THREE = root.THREE, H = root.Helio, K = H.lib, D = H.D;
  const PI = Math.PI, TAU = PI * 2;

  // ───────────────────────────── BSP CSG (after Evan Wallace's csg.js) ─────
  const EPS = 1e-5;
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  class Poly {
    constructor(v, pl) { this.v = v; this.pl = pl; }
    flip() { this.v.reverse(); this.pl = { n: [-this.pl.n[0], -this.pl.n[1], -this.pl.n[2]], w: -this.pl.w }; }
  }
  function split(pl, p, cf, cb, f, b) {
    let type = 0; const ts = [];
    for (const v of p.v) { const t = dot(pl.n, v) - pl.w; const ty = t < -EPS ? 2 : t > EPS ? 1 : 0; type |= ty; ts.push(ty); }
    if (type === 0) (dot(pl.n, p.pl.n) > 0 ? cf : cb).push(p);
    else if (type === 1) f.push(p);
    else if (type === 2) b.push(p);
    else {
      const fv = [], bv = [], n = p.v.length;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, ti = ts[i], tj = ts[j], vi = p.v[i], vj = p.v[j];
        if (ti !== 2) fv.push(vi);
        if (ti !== 1) bv.push(vi);
        if ((ti | tj) === 3) {
          const t = (pl.w - dot(pl.n, vi)) / dot(pl.n, sub(vj, vi));
          const v = lerp3(vi, vj, t); fv.push(v); bv.push(v);
        }
      }
      if (fv.length >= 3) f.push(new Poly(fv, { n: p.pl.n.slice(), w: p.pl.w }));
      if (bv.length >= 3) b.push(new Poly(bv, { n: p.pl.n.slice(), w: p.pl.w }));
    }
  }
  class Node {
    constructor(ps) { this.pl = null; this.f = null; this.b = null; this.ps = []; if (ps) this.build(ps); }
    invert() {
      const st = [this];
      while (st.length) {
        const n = st.pop();
        n.ps.forEach(p => p.flip());
        if (n.pl) n.pl = { n: [-n.pl.n[0], -n.pl.n[1], -n.pl.n[2]], w: -n.pl.w };
        const t = n.f; n.f = n.b; n.b = t;
        if (n.f) st.push(n.f); if (n.b) st.push(n.b);
      }
    }
    clipPolygons(ps) {
      if (!this.pl) return ps.slice();
      let f = [], b = [];
      for (const p of ps) split(this.pl, p, f, b, f, b);
      if (this.f) f = this.f.clipPolygons(f);
      b = this.b ? this.b.clipPolygons(b) : [];
      return f.concat(b);
    }
    clipTo(bsp) {
      const st = [this];
      while (st.length) { const n = st.pop(); n.ps = bsp.clipPolygons(n.ps); if (n.f) st.push(n.f); if (n.b) st.push(n.b); }
    }
    all() {
      const out = [], st = [this];
      while (st.length) { const n = st.pop(); out.push(...n.ps); if (n.f) st.push(n.f); if (n.b) st.push(n.b); }
      return out;
    }
    build(ps) {
      const st = [[this, ps]];
      while (st.length) {
        const [n, list] = st.pop();
        if (!list.length) continue;
        if (!n.pl) n.pl = { n: list[0].pl.n.slice(), w: list[0].pl.w };
        const f = [], b = [];
        for (const p of list) split(n.pl, p, n.ps, n.ps, f, b);
        if (f.length) { if (!n.f) n.f = new Node(); st.push([n.f, f]); }
        if (b.length) { if (!n.b) n.b = new Node(); st.push([n.b, b]); }
      }
    }
  }
  function toPolys(shell) {
    const P = shell.p, out = [];
    for (let i = 0; i < P.length; i += 9) {
      const a = [P[i], P[i + 1], P[i + 2]], b = [P[i + 3], P[i + 4], P[i + 5]], c = [P[i + 6], P[i + 7], P[i + 8]];
      const n = cross(sub(b, a), sub(c, a)), l = Math.hypot(n[0], n[1], n[2]);
      if (l < 1e-10) continue;
      const nn = [n[0] / l, n[1] / l, n[2] / l];
      out.push(new Poly([a, b, c], { n: nn, w: dot(nn, a) }));
    }
    return out;
  }
  const clonePs = ps => ps.map(p => new Poly(p.v.slice(), { n: p.pl.n.slice(), w: p.pl.w }));
  const csgSub = (A, B) => { A = clonePs(A); B = clonePs(B); const a = new Node(A), b = new Node(B); a.invert(); a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.all()); a.invert(); return a.all(); };
  const csgUnion = (A, B) => { A = clonePs(A); B = clonePs(B); const a = new Node(A), b = new Node(B); a.clipTo(b); b.clipTo(a); b.invert(); b.clipTo(a); b.invert(); a.build(b.all()); return a.all(); };
  const csgAnd = (A, B) => { A = clonePs(A); B = clonePs(B); const a = new Node(A), b = new Node(B); a.invert(); b.clipTo(a); b.invert(); a.clipTo(b); b.clipTo(a); a.build(b.all()); a.invert(); return a.all(); };

  // polygons → closed shell; T-junctions left by the BSP are stitched so every edge pairs up
  function toShell(polys) {
    const kf = v => Math.round(v[0] * 1e3) + ',' + Math.round(v[1] * 1e3) + ',' + Math.round(v[2] * 1e3);
    const idx = new Map(), V = [], F = [];
    const vid = v => { const k = kf(v); let i = idx.get(k); if (i === undefined) { i = V.length; V.push(v); idx.set(k, i); } return i; };
    for (const p of polys) {
      const ids = p.v.map(vid).filter((x, i, a) => x !== a[(i + 1) % a.length]);
      if (ids.length >= 3) F.push(ids);
    }
    const CELL = 6, grid = new Map(), gk = (x, y, z) => x + ',' + y + ',' + z;
    V.forEach((v, i) => {
      const k = gk(Math.floor(v[0] / CELL), Math.floor(v[1] / CELL), Math.floor(v[2] / CELL));
      let a = grid.get(k); if (!a) grid.set(k, a = []); a.push(i);
    });
    const near = (a, b) => {
      const A = V[a], B = V[b], d = sub(B, A), len = Math.hypot(d[0], d[1], d[2]);
      const out = [];
      if (len < 1e-6) return out;
      const steps = Math.max(1, Math.ceil(len / CELL)), seen = new Set();
      for (let s = 0; s <= steps; s++) {
        const q = lerp3(A, B, s / steps), cx = Math.floor(q[0] / CELL), cy = Math.floor(q[1] / CELL), cz = Math.floor(q[2] / CELL);
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
          const arr = grid.get(gk(cx + dx, cy + dy, cz + dz)); if (!arr) continue;
          for (const i of arr) {
            if (i === a || i === b || seen.has(i)) continue; seen.add(i);
            const w = sub(V[i], A), t = dot(w, d) / (len * len);
            if (t <= 1e-6 || t >= 1 - 1e-6) continue;
            const c = cross(w, d);
            if (Math.hypot(c[0], c[1], c[2]) / len < 2e-4) out.push([t, i]);
          }
        }
      }
      return out.sort((x, y) => x[0] - y[0]).map(x => x[1]);
    };
    const tris = [];
    const emit = (a, b, c) => {
      if (a === b || b === c || a === c) return;
      const x = cross(sub(V[b], V[a]), sub(V[c], V[a]));
      if (Math.hypot(x[0], x[1], x[2]) > 1e-7) tris.push([a, b, c]);
    };
    for (const f of F) {
      const ring = [];
      for (let i = 0; i < f.length; i++) { ring.push(f[i]); ring.push(...near(f[i], f[(i + 1) % f.length])); }
      if (ring.length === f.length) {
        let ok = true;
        for (let i = 1; i < f.length - 1 && ok; i++) { const c = cross(sub(V[f[i]], V[f[0]]), sub(V[f[i + 1]], V[f[0]])); if (Math.hypot(c[0], c[1], c[2]) < 1e-6) ok = false; }
        if (ok) { for (let i = 1; i < f.length - 1; i++) emit(f[0], f[i], f[i + 1]); continue; }
      }
      const c = [0, 0, 0]; f.forEach(i => { c[0] += V[i][0] / f.length; c[1] += V[i][1] / f.length; c[2] += V[i][2] / f.length; });
      const ci = V.length; V.push(c);
      for (let i = 0; i < ring.length; i++) emit(ci, ring[i], ring[(i + 1) % ring.length]);
    }
    // drop coincident back-to-back pairs (zero-thickness skins left where two solids touched)
    const seen = new Map();
    tris.forEach((t, i) => { const k = t.slice().sort((a, b) => a - b).join(','); (seen.get(k) || seen.set(k, []).get(k)).push(i); });
    const drop = new Set();
    for (const list of seen.values()) if (list.length >= 2) {
      const sig = i => { const t = tris[i], m = t.indexOf(Math.min(...t)); return t[(m + 1) % 3] < t[(m + 2) % 3] ? 1 : -1; };
      const pos = list.filter(i => sig(i) > 0), neg = list.filter(i => sig(i) < 0);
      for (let k = 0; k < Math.min(pos.length, neg.length); k++) { drop.add(pos[k]); drop.add(neg[k]); }
      pos.slice(neg.length).slice(1).forEach(i => drop.add(i)); neg.slice(pos.length).slice(1).forEach(i => drop.add(i));
    }
    let T2 = tris.filter((_, i) => !drop.has(i));
    // repair: drop isolated flakes, then close any pinholes left by float round-off
    const edges = list => { const m = new Map(); list.forEach(t => { for (let k = 0; k < 3; k++) { const e = t[k] + '>' + t[(k + 1) % 3]; m.set(e, (m.get(e) || 0) + 1); } }); return m; };
    const open = (m, a, b) => (m.get(a + '>' + b) || 0) > (m.get(b + '>' + a) || 0);
    let m = edges(T2);
    T2 = T2.filter(t => !(open(m, t[0], t[1]) && open(m, t[1], t[2]) && open(m, t[2], t[0])));
    m = edges(T2);
    const next = new Map();
    for (const [e, n] of m) { const [a, b] = e.split('>').map(Number); if (open(m, a, b)) (next.get(a) || next.set(a, []).get(a)).push(b); }
    for (const [start] of next) {
      const loop = [start];
      let cur = start, guard = 0;
      while (guard++ < 10000) {
        const outs = next.get(cur); if (!outs || !outs.length) break;
        const nx = outs.pop(); if (nx === start) break;
        loop.push(nx); cur = nx;
      }
      for (let i = 1; i < loop.length - 1; i++) T2.push([loop[0], loop[i + 1], loop[i]]);
    }
    const s = new K.Shell();
    T2.forEach(t => s.tri(V[t[0]], V[t[1]], V[t[2]]));
    return s.orient();
  }
  function bbox(sh) {
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9], P = sh.p;
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], P[i + k]); mx[k] = Math.max(mx[k], P[i + k]); }
    return [mn, mx];
  }
  function polyBox(p) {
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (const v of p.v) for (let k = 0; k < 3; k++) { if (v[k] < mn[k]) mn[k] = v[k]; if (v[k] > mx[k]) mx[k] = v[k]; }
    return [mn, mx];
  }
  const overlap = (a, b) => [0, 1, 2].every(k => a[0][k] <= b[1][k] + 0.05 && b[0][k] <= a[1][k] + 0.05);

  // ───────────────────────────── cutters ───────────────────────────────────
  // cylinder along axis 'x' | 'y' | 'z' through point p, from a0 to a1 along that axis
  function onAxis(shell, axis, p) {
    const m = new THREE.Matrix4();
    if (axis === 'x') m.makeRotationZ(-PI / 2).premultiply(new THREE.Matrix4().makeTranslation(0, p[1], p[2]));
    else if (axis === 'z') m.makeRotationX(PI / 2).premultiply(new THREE.Matrix4().makeTranslation(p[0], p[1], 0));
    else m.makeTranslation(p[0], 0, p[2]);
    return K.transformShell(shell, m);
  }
  const cyl = (axis, p, a0, a1, r, n = 28) => onAxis(K.radialSolid([{ y: a0, r }, { y: a1, r }], n), axis, p);
  const cone = (axis, p, a0, a1, r0, r1, n = 28) => onAxis(K.radialSolid([{ y: a0, r: r0 }, { y: a1, r: r1 }], n), axis, p);
  const boxS = (x0, x1, y0, y1, z0, z1) => K.box(x0, x1, y0, y1, z0, z1);
  const ellipse = (a, b, n = 72) => Array.from({ length: n }, (_, i) => [a * Math.cos(i / n * TAU), b * Math.sin(i / n * TAU)]);
  const placeM = p => {
    const m = new THREE.Matrix4();
    const e = new THREE.Euler(...(p.rot || [0, 0, 0]), p.rotOrder || 'XYZ');
    return m.compose(new THREE.Vector3(...(p.pos || [0, 0, 0])), new THREE.Quaternion().setFromEuler(e), new THREE.Vector3(1, 1, 1));
  };

  async function buildKit(displayParts, onProgress) {
    const byId = Object.fromEntries(displayParts.map(p => [p.id, p]));
    const T = D.T, L = D.L, P = D.PED_TOP, PB = D.PLATE_BOT, B = D.BASE_TOP;
    const CW = D.CW, CZ0 = D.CZ0, CZ1 = D.CZ1, DX = CW - T - 1, DI = DX - T;
    const xm = z => (D.hwOut(z) + D.hwIn(z)) / 2;
    const cuts = {}, fast = [];
    const cut = (id, frame, make) => (cuts[id] = cuts[id] || []).push({ frame, make });
    const hole = (id, frame, axis, p, a0, a1, r) => cut(id, frame, () => cyl(axis, p, a0, a1, r));
    const F = (kind, frame, axis, p, a0, a1, r, withIds, label) => fast.push({ kind, frame, axis, p, a0, a1, r, withIds, label });

    // joint recipes ------------------------------------------------------
    const R_DOWEL = 2.6, R_INSERT = 2.0, R_CLEAR = 1.7, R_HEAD = 3.1;
    function screwUp(frame, thru, into, x, z, y0, y1, len, label) {
      hole(thru, frame, 'y', [x, 0, z], y0 - 0.1, y1 + 0.1, R_CLEAR);
      hole(thru, frame, 'y', [x, 0, z], y0 - 0.1, y0 + 3.5, R_HEAD);
      hole(into, frame, 'y', [x, 0, z], y1 - 0.1, y1 + 6.6, R_INSERT);
      F('screw', frame, 'y', [x, 0, z], y0 + 3.5, y0 + 3.5 + len, 1.5, [thru, into], label);
      F('insert', frame, 'y', [x, 0, z], y1, y1 + 5.7, 2.1, [into], 'M3 heat-set insert (M3 × 5.7, 4.0 mm hole)');
    }
    function dowel(frame, axis, p, lower, upper, j, withIds) {
      const a = lower, b = upper;
      hole(a, frame, axis, p, j - 8.1, j + 0.1, R_DOWEL);
      hole(b, frame, axis, p, j - 0.1, j + 8.1, R_DOWEL);
      F('dowel', frame, axis, p, j - 7.8, j + 7.8, 2.45, withIds || [a, b], 'Dowel pin Ø4.9 × 15.6 (printed, or 5 mm beech dowel)');
    }
    // a dowel across a split seam of one part: one cutter, both halves get their half
    const seamDowel = (id, frame, axis, p) => { hole(id, frame, axis, p, -8.1, 8.1, R_DOWEL); F('dowel', frame, axis, p, -7.8, 7.8, 2.45, [id], 'Dowel pin Ø4.9 × 15.6 (printed, or 5 mm beech dowel)'); };

    // ── viewer body ──
    const V = 'viewer';
    [[1, 30], [1, 145], [-1, 30], [-1, 145]].forEach(([s, z]) => screwUp(V, 'floor', s > 0 ? 'wall-r' : 'wall-l', s * xm(z), z, 0, D.FLOOR_TOP, 16, 'M3 × 16 socket-head screw'));
    [[50, 6.35, 'lens-board'], [-50, 6.35, 'lens-board'], [60, L - 6.35, 'end-wall'], [-60, L - 6.35, 'end-wall']].forEach(([x, z, id]) =>
      screwUp(V, 'floor', id, x, z, 0, D.FLOOR_TOP, 16, 'M3 × 16 socket-head screw'));
    [[1, 30], [1, 145], [-1, 30], [-1, 145]].forEach(([s, z]) => dowel(V, 'y', [s * xm(z), 0, z], s > 0 ? 'wall-r' : 'wall-l', 'roof', D.WALL_TOP));
    dowel(V, 'y', [0, 0, 6.35], 'lens-board', 'roof', D.WALL_TOP);
    dowel(V, 'y', [0, 0, L - 6.35], 'end-wall', 'roof', D.WALL_TOP);
    [['lens-board', 6.35], ['end-wall', L - 6.35]].forEach(([board, zc]) => [1, -1].forEach(s => [35, 90].forEach(y => {
      const xj = D.hwIn(zc), wall = s > 0 ? 'wall-r' : 'wall-l';
      if (s > 0) dowel(V, 'x', [0, y, zc], board, wall, xj);
      else dowel(V, 'x', [0, y, zc], wall, board, -xj);
    })));
    cut('lens-board', V, () => boxS(-1.65, 1.65, 14, 92.5, 8, 12.8));                        // septum slot
    [50, -50].forEach(x => {                                                                // baffle → roof
      hole('baffle', V, 'y', [x, 0, 30.5], 103.9, 110.1, 1.6); hole('roof', V, 'y', [x, 0, 30.5], 109.9, 114.1, 1.6);
      F('pin', V, 'y', [x, 0, 30.5], 104.3, 113.7, 1.45, ['baffle', 'roof'], 'Pin Ø2.9 × 9.4 (printed, or 3 mm rod)');
    });
    cut('baffle', V, () => boxS(-1.9, 1.9, 85, 92.8, 25, 36));                                // clears the septum
    [1, -1].forEach(s => [50, 150].forEach(z => {                                            // lid magnets
      const x = s * (D.hwOut(z) - 6.4);
      hole('roof', V, 'y', [x, 0, z], D.ROOF_TOP - 2.3, D.ROOF_TOP + 0.1, 3.1);
      hole('lid', 'lid', 'y', [x, 0, z - D.LID_REAR], -0.1, 2.2, 3.1);
      F('magnet', V, 'y', [x, 0, z], D.ROOF_TOP - 2, D.ROOF_TOP, 3, ['roof'], 'Magnet 6 × 2 mm (N52)');
      F('magnet', 'lid', 'y', [x, 0, z - D.LID_REAR], 0, 2, 3, ['lid'], 'Magnet 6 × 2 mm (N52)');
    }));
    const SUN_Z = (D.LID_FRONT - D.LID_REAR) / 2;
    cut('lid', 'lid', () => boxS(-75.3, 75.3, D.LID_H - 1.4, D.LID_H + 0.5, SUN_Z - 50.3, SUN_Z + 50.3));
    [60, -60].forEach(x => {                                                                  // crest → roof
      const p = [x, 0, 179.5];
      hole('rest', V, 'y', p, D.ROOF_TOP - 0.1, D.ROOF_TOP + 4.1, R_CLEAR);
      cut('rest', V, () => cone('y', p, D.ROOF_TOP + 2.2, D.ROOF_TOP + 4.1, 1.7, 3.6));
      hole('roof', V, 'y', p, D.ROOF_TOP - 6.6, D.ROOF_TOP + 0.1, R_INSERT);
      F('screw', V, 'y', p, D.ROOF_TOP - 6, D.ROOF_TOP + 4, 1.5, ['rest', 'roof'], 'M3 × 10 countersunk screw');
      F('insert', V, 'y', p, D.ROOF_TOP - 5.7, D.ROOF_TOP, 2.1, ['roof'], 'M3 heat-set insert (M3 × 5.7, 4.0 mm hole)');
    });
    [[82, 26], [-82, 26], [82, 95.5], [-82, 95.5]].forEach(([x, y]) => {                     // ground-glass frame
      hole('end-frame', V, 'z', [x, y, 0], L - 0.1, L + 3.2, 1.6); hole('end-wall', V, 'z', [x, y, 0], L - 3.2, L + 0.1, 1.6);
      F('pin', V, 'z', [x, y, 0], L - 3, L + 2.9, 1.45, ['end-frame', 'end-wall'], 'Pin Ø2.9 × 5.9 (printed, or 3 mm rod)');
    });
    [['medal-r', 'wall-r'], ['medal-l', 'wall-l'], ['base-medal-r', 'base-side-r'], ['base-medal-l', 'base-side-l']].forEach(([m, w]) => {
      const M = placeM(byId[m]);
      cut(w, byId[w].parent, () => K.transformShell(K.extrudePlan(ellipse(30.4, 40.4), [], -1.0, 1.05), M));
    });
    // bayonet socket: flanged ring captured under a screwed cap
    const HZ = D.HUB_Z;
    cut('floor', V, () => cyl('y', [0, 0, HZ], 4.4, 13.0, 64.3, 96));
    cut('floor', V, () => cyl('y', [0, 0, HZ], 9.6, 13.0, 70.3, 96));
    [30, 150, 270].forEach(deg => {
      const a = deg * PI / 180, p = [67 * Math.cos(a), 0, HZ + 67 * Math.sin(a)];
      hole('socket-cap', V, 'y', p, 9.5, 14.1, R_CLEAR); hole('socket-cap', V, 'y', p, 11.0, 14.1, R_HEAD);
      hole('floor', V, 'y', p, 3.0, 9.7, R_INSERT);
      F('screw', V, 'y', p, 3.2, 11.0, 1.5, ['socket-cap', 'floor'], 'M3 × 8 socket-head screw');
      F('insert', V, 'y', p, 3.3, 9.0, 2.1, ['floor'], 'M3 heat-set insert (M3 × 5.7, 4.0 mm hole)');
    });
    [240, 0, 120].forEach(deg => {
      const a = deg * PI / 180, p = [38 * Math.cos(a), 0, HZ + 38 * Math.sin(a)];
      hole('socket-cap', V, 'y', p, 9.5, 12.8, 3.1);
      F('magnet', V, 'y', p, 9.6, 12.6, 3, ['socket-cap'], 'Magnet 6 × 3 mm (N52)');
    });

    // optional split floor and roof for beds under 256 mm: the same cuts plus seam dowel holes
    const altCuts = (id, extra) => { (cuts[id] = cuts[id] || []); extra.forEach(([ax, pt]) => hole(id, V, ax, pt, -8.1, 8.1, R_DOWEL)); };
    altCuts('floor-alt', [['x', [0, 6.35, 4]], ['x', [0, 6.35, 172]]]);
    altCuts('roof-alt', [['x', [0, D.ROOF_TOP - 6.35, 20]], ['x', [0, D.ROOF_TOP - 6.35, 180]]]);

    // ── pedestal ──
    const PD = 'pedestal';
    const sideX = CW - 6.35, backZ = CZ0 + 6.35;
    const carcase = [[sideX, -90, 'base-side-r'], [sideX, 70, 'base-side-r'], [-sideX, -90, 'base-side-l'], [-sideX, 70, 'base-side-l'], [60, backZ, 'base-back'], [-60, backZ, 'base-back']];
    carcase.forEach(([x, z, id]) => { screwUp(PD, 'plinth', id, x, z, 0, 16, 16, 'M3 × 16 socket-head screw'); dowel(PD, 'y', [x, 0, z], id, 'base-top', 121); });
    [1, -1].forEach(s => [40, 100].forEach(y => {
      const j = s * (CW - T), side = s > 0 ? 'base-side-r' : 'base-side-l';
      if (s > 0) dowel(PD, 'x', [0, y, backZ], 'base-back', side, j); else dowel(PD, 'x', [0, y, backZ], side, 'base-back', j);
    }));
    [[65, 'z'], [-65, 'z'], [65, 'x'], [-65, 'x']].forEach(([c, ax]) => {
      const p = ax === 'x' ? [0, 8, c] : [c, 8, 0], q = ax === 'x' ? [0, 128.5, c] : [c, 128.5, 0];
      seamDowel('plinth', PD, ax, p); seamDowel('base-top', PD, ax, q);
    });
    hole('base-top', PD, 'y', [0, 0, 0], 120.9, B + 0.1, 4.4);                               // M8 rod
    [[28.3, 28.3], [-28.3, -28.3]].forEach(([x, z]) => dowel(PD, 'y', [x, 0, z], 'base-top', 'column-foot', B, ['base-top', 'column']));
    [80, -80].forEach(z => seamDowel('top-plate', PD, 'x', [0, PB + 12, z]));
    cut('top-plate', PD, () => cone('y', [0, 0, 0], P - 7.05, P + 0.05, 45.3, 52.35, 120));
    [0, 120, 240].forEach(deg => {
      const a = deg * PI / 180, p = [30 * Math.cos(a), 0, 30 * Math.sin(a)];
      hole('boss-core', PD, 'y', p, P - 1.3, P + 5.4, R_INSERT);
      F('screw', PD, 'y', p, P - 1, P + 7, 1.5, ['boss-core', 'lug-ring'], 'M3 × 8 button-head screw');
      F('insert', PD, 'y', p, P - 0.4, P + 5.3, 2.1, ['boss-core'], 'M3 heat-set insert (M3 × 5.7, 4.0 mm hole)');
    });
    [60, 180, 300].forEach(deg => { const a = deg * PI / 180; F('magnet', PD, 'y', [38 * Math.cos(a), 0, 38 * Math.sin(a)], P + 7.2, P + 9.2, 3, ['lug-ring'], 'Magnet 6 × 2 mm (N52)'); });
    F('rod', PD, 'y', [0, 0, 0], 111, P + 4.5, 4, ['column', 'boss-core'], 'M8 threaded rod, 260 mm');
    F('nut', PD, 'y', [0, 0, 0], P - 1.6, P + 4.9, 6.5, ['boss-core'], 'M8 nut');
    F('nut', PD, 'y', [0, 0, 0], 113.5, 120, 6.5, ['base-top'], 'M8 nut');
    F('nut', PD, 'y', [0, 0, 0], 120, 121, 8, ['base-top'], 'M8 washer, 16 mm');

    // ── drawer ──
    const DR = 'drawer';
    cut('drawer-side-r', DR, () => boxS(DI - 0.1, DI + 3.0, 18.8, 25.2, -107.1, CZ1 + 0.1));
    cut('drawer-side-l', DR, () => boxS(-DI - 3.0, -DI + 0.1, 18.8, 25.2, -107.1, CZ1 + 0.1));
    cut('drawer-back', DR, () => boxS(-DI - 0.1, DI + 0.1, 18.8, 25.2, -97.3, -94.2));
    cut('drawer-front', DR, () => boxS(-DI - 3.1, DI + 3.1, 18.8, 25.2, CZ1 - 0.1, CZ1 + 3.0));
    [1, -1].forEach(s => [45, 100].forEach(y => {
      const side = s > 0 ? 'drawer-side-r' : 'drawer-side-l';
      dowel(DR, 'z', [s * (DX - 6.35), y, 0], side, 'drawer-front', CZ1);
      if (s > 0) dowel(DR, 'x', [0, y, -100.65], 'drawer-back', side, DI); else dowel(DR, 'x', [0, y, -100.65], side, 'drawer-back', -DI);
    }));
    hole('drawer-front', DR, 'z', [0, 68.5, 0], CZ1 - 0.1, D.DF + 0.1, R_CLEAR);
    hole('drawer-front', DR, 'z', [0, 68.5, 0], CZ1 - 0.1, CZ1 + 3.5, R_HEAD);
    F('screw', DR, 'z', [0, 68.5, 0], CZ1 + 3.5, CZ1 + 28.5, 1.5, ['drawer-front', 'knob'], 'M3 × 25 pan-head screw (knob)');

    // ── kit parts -------------------------------------------------------
    const kit = [];
    const from = (id, extra = {}) => { const d = byId[id]; return Object.assign({ id: id, name: d.name, group: d.group, mat: d.mat, qty: d.qty, parent: d.parent, from: [id],
      pos: d.pos, rot: d.rot, rotOrder: d.rotOrder, explode: d.explode, print: d.print, down: d.down, pair: d.pair, multi: d.multi, shared: d.shared,
      base: () => d.shells() }, extra); };
    const bodyPoly = [[-D.HW0, 0], [D.HW0, 0], [D.HW1, L], [-D.HW1, L]];
    const same = 'Unchanged from the final design.';
    kit.push(
      from('floor', { note: 'Counterbored for 8 M3 screws from below. Stepped seat for the flanged bayonet socket and its cap, with 3 insert holes. Print bottom-down.' }),
      from('wall-r', { note: '2 insert holes underneath, 2 dowel holes on top, 4 dowel holes for the boards, 1 mm medallion recess. Prints lying on its outer face.' }),
      from('wall-l', { note: 'Mirror of the right wall.' }),
      from('lens-board', { note: '2 inserts below, 1 dowel on top, 4 end dowels, slot for the septum. Prints outer face down.' }),
      from('end-wall', { note: '2 inserts below, 1 dowel on top, 4 end dowels, 4 pin holes for the frame. Prints outer face down.' }),
      from('floor', { id: 'floor-alt', name: 'Floor board, split', alt: true, from: [], split: 'half', extraCuts: 'floor',
        note: 'Only if your bed is under 256 mm: the floor in two halves with 2 seam dowels. Glue the seam.' }),
      from('roof', { id: 'roof-alt', name: 'Roof, split', alt: true, from: [], split: 'half', extraCuts: 'roof',
        note: 'Only if your bed is under 256 mm: the roof in two halves with 2 seam dowels. Glue the seam; the dowels on the walls keep it square.' }),
      from('roof', { note: '6 dowel holes underneath (lift-off for access), 4 lid-magnet pockets, 2 inserts for the crest, 2 baffle pin holes. Prints bottom-down: the glass ledge steps out, never over air.' }),
      { id: 'cornice', name: 'Cornice moulding', group: 'Ornament', mat: 'gilt', qty: 1, parent: V, from: ['cornice'], explode: byId.cornice.explode,
        note: 'Opened 0.25 mm so it slides over the walls; the roof overhang traps it. A few dots of CA. The cove leans out at 32°, so it prints without support.',
        base: () => [K.sweepRing(bodyPoly, [[0.25, 101], [0.5, 101], [1.8, 102.4], [2.5, 104.2], [3.3, 105.8], [4.6, 107.1], [5.7, 108.2], [6.1, 110], [0.25, 110]])] },
      { id: 'gate-front', name: 'Card gate, front', group: 'Optics', mat: 'black', qty: 1, parent: V, from: ['gate'], explode: byId.gate.explode,
        note: 'Split from the back plate so neither half overhangs. Print front face down. Gate, glass and back plate are trapped between the walls; no fasteners.',
        base: () => byId.gate.shells().slice(0, 2) },
      { id: 'gate-back', name: 'Card gate, back plate', group: 'Optics', mat: 'black', qty: 1, parent: V, from: [], explode: [0, 0, 52],
        note: 'Glue to the gate front, or leave loose; it is trapped.', base: () => byId.gate.shells().slice(2) },
      { id: 'septum', name: 'Septum', group: 'Optics', mat: 'black', qty: 1, parent: V, from: ['septum'], explode: byId.septum.explode,
        note: 'Lengthened 4.5 mm to key into the slot in the lens board. Glue.', base: () => [K.box(-1.5, 1.5, 14.5, 92, 8.2, 120)] },
      { id: 'baffle', name: 'Glare baffle', group: 'Optics', mat: 'black', qty: 1, parent: V, from: ['baffle'], explode: byId.baffle.explode, down: [0, 1, 0],
        note: 'Given a T-flange so two pins can hold it to the roof. Print flange down.',
        base: () => [K.extrudeZY([[26, 86], [29, 86], [29, 104], [35, 104], [35, 110], [26, 110]], -76, 76)] },
      { id: 'socket-ring', name: 'Bayonet socket ring', group: 'Mount', mat: 'brass', qty: 1, parent: V, from: ['socket'], pos: [0, 0, HZ], explode: [0, -130, 0], print: 'axis',
        note: 'Split from its cap so the gallery never bridges. The chamfered flange drops into the floor’s counterbore. Print lip down.',
        base: () => {
          const lugHalf = 11 * PI / 180;
          const ang = (a, b) => { let d = (a - b) % TAU; if (d > PI) d -= TAU; if (d < -PI) d += TAU; return d; };
          const LUGS = [0, TAU / 3, 2 * TAU / 3];
          const lip = th => LUGS.some(a => Math.abs(ang(th, a + PI + D.LOCK_TURN)) < lugHalf) ? 53.5 : 46;
          const gal = th => LUGS.some(a => { const d = ang(th, a + PI); return d > -lugHalf && d < D.LOCK_TURN + lugHalf; }) ? 53.5 : 46;
          return [K.tubeSolid([{ y: 0, ro: 62, ri: lip }, { y: 4.6, ro: 62, ri: lip }, { y: 5, ro: 62.4, ri: lip }, { y: 5, ro: 62.4, ri: gal },
            { y: 6.6, ro: 64, ri: gal }, { y: 9.6, ro: 64, ri: gal }], 360)];
        } },
      { id: 'socket-cap', name: 'Bayonet socket cap', group: 'Mount', mat: 'brass', qty: 1, parent: V, from: [], pos: [0, 0, HZ], explode: [0, -95, 0],
        note: 'Screws down over the ring flange with 3 M3 screws; 3 magnets sit in its underside.',
        base: () => [K.radialSolid([{ y: 9.6, r: 70 }, { y: 13.2, r: 70 }, { y: 14, r: 69.2 }], 120)] },
      from('end-frame', { note: '4 pin holes in the back. Prints back-down.' }),
      from('lid', { note: '4 magnet pockets underneath and a 1.4 mm pocket that locates the Sun. Prints bottom-down.' }),
      from('sun', { note: same + ' Glue into the lid pocket.' }),
      from('rest', { down: [0, -1, 0], note: '2 countersunk holes in the groove floor (hidden by the standing lid). Prints as installed.' }),
      from('medal-r', { id: 'medal', from: ['medal-r', 'medal-l', 'base-medal-r', 'base-medal-l'], qty: 4, shared: 'medal', note: same + ' Glue into the 1 mm recesses.' }),
      from('sleeve', { note: 'Rosette flange is now a 45° cone. Print outer face down. Held by the lock clip behind the board.' }),
      { id: 'sleeve-clip', name: 'Sleeve lock clip', group: 'Eyepieces', mat: 'black', qty: 2, parent: V, from: [], pair: true, explode: [0, 0, 40],
        note: 'Snaps onto the sleeve barrel inside the box so the eyepiece can’t pull out.',
        base: () => {
          const pts = [];
          for (let i = 0; i <= 60; i++) { const a = PI / 2 + (25 + 310 * i / 60) * PI / 180; pts.push([30 * Math.cos(a), 30 * Math.sin(a)]); }
          for (let i = 60; i >= 0; i--) { const a = PI / 2 + (25 + 310 * i / 60) * PI / 180; pts.push([25.05 * Math.cos(a), 25.05 * Math.sin(a)]); }
          return [K.extrudeXY(pts, [], 12.8, 17.8)];
        } },
      from('draw-tube', { note: 'Grip now steps out at 45°. Print eyecup down; the 3-start thread needs no support.' }),
      from('lens-ring', { note: same }),
      from('plinth', { split: 'quad', note: 'Quartered with seam dowels to fit a 256 mm bed. Counterbored for 6 M3 screws from below.' }),
      from('base-side-r', { note: 'Inserts below, dowels on top, dowel holes for the back, medallion recess.' }),
      from('base-side-l', { note: 'Mirror of the right side.' }),
      from('base-back', { note: 'Inserts below, dowels on top and at both ends.' }),
      from('base-top', { split: 'quad', note: 'Quartered with seam dowels. M8 rod hole, 2 column dowels, 6 carcase dowels.' }),
      { id: 'column', name: 'Gilt column', group: 'Pedestal', mat: 'gilt', qty: 1, parent: PD, from: ['column'], explode: byId.column.explode, print: 'axis',
        note: 'Bored 8.6 mm for the M8 rod; 2 dowel holes in the foot stop it turning. Every flare is ≤ 45°: print upright, no support.',
        base: () => {
          const foot = K.tubeSolid(K.columnRings(B, B + 8.1).map(r => ({ y: r.y, ro: r.r, ri: 4.3 })), 96);
          const body = K.tubeSolid(K.columnRings(B + 7.9, PB).map(r => ({ y: r.y, ro: r.r, ri: 4.3 })), 192);
          return [foot, body];
        }, cutShell: 0 },
      from('console-0', { id: 'console', from: ['console-0', 'console-1', 'console-2', 'console-3', 'console-4', 'console-5', 'console-6', 'console-7'], qty: 8, shared: 'console',
        note: same + ' Glue (epoxy) to the column, base and top; they carry no load.' }),
      { id: 'top-plate', name: 'Pedestal top', group: 'Pedestal', mat: 'walnut', qty: 1, parent: PD, from: ['top-plate'], explode: byId['top-plate'].explode, split: 'half',
        note: 'Halved with seam dowels. Its countersunk centre hole is clamped between the column and the boss core by the M8 rod.',
        base: () => [K.sweepRing(D.PLATE_POLY, [[-7, 0], [-6, 0], [-3, 1.5], [-1, 4], [0, 8], [0, 16], [-1, 20], [-3, 22.5], [-6, 24], [-7, 24]].map(([o, y]) => [o, PB + y])),
          K.extrudePlan(K.insetPoly(D.PLATE_POLY, 7), [K.circle(0, 0, 45.3, 120)], PB, 24)] },
      { id: 'boss-core', name: 'Bayonet boss core', group: 'Mount', mat: 'brass', qty: 1, parent: PD, from: ['boss'], explode: [0, 125, 0], print: 'axis',
        note: 'A 45° cone clamps the top plate. Hex trap for the M8 nut on top. Prints upright.',
        base: () => {
          const hex = th => 6.65 / Math.cos((((th + PI / 6) % (PI / 3)) + PI / 3) % (PI / 3) - PI / 6);
          return [K.tubeSolid([{ y: PB, ro: 45, ri: 4.3 }, { y: P - 7, ro: 45, ri: 4.3 }, { y: P - 1.7, ro: 50.3, ri: 4.3 }, { y: P - 1.7, ro: 50.3, ri: hex },
            { y: P, ro: 52, ri: hex }, { y: P, ro: 45, ri: hex }, { y: P + 5.3, ro: 45, ri: hex }], 96)];
        } },
      { id: 'lug-ring', name: 'Bayonet lug ring', group: 'Mount', mat: 'brass', qty: 1, parent: PD, from: [], explode: [0, 150, 0],
        note: 'Flat, so the lugs print without support. 3 button-head screws hold it to the core and cover the nut; 3 magnets on top.',
        base: () => {
          const LUGS = [0, TAU / 3, 2 * TAU / 3], half = 8.8 * PI / 180;
          const ang = (a, b) => { let d = (a - b) % TAU; if (d > PI) d -= TAU; if (d < -PI) d += TAU; return d; };
          const out = [];
          for (let i = 0; i < 360; i++) {
            const th = i / 360 * TAU, lug = LUGS.some(a => Math.abs(ang(th, a)) < half);
            out.push([(lug ? 52.5 : 45) * Math.cos(th), (lug ? 52.5 : 45) * Math.sin(th)]);
          }
          const at = (r, deg, rr) => K.circle(r * Math.cos(deg * PI / 180), r * Math.sin(deg * PI / 180), rr, 24);
          return [K.extrudePlan(out, [0, 120, 240].map(d => at(30, d, 1.7)), P + 5.3, 1.75),
            K.extrudePlan(out, [...[0, 120, 240].map(d => at(30, d, 3.2)), ...[60, 180, 300].map(d => at(38, d, 3.1))], P + 7.0, 2.2)];
        } },
      from('drawer-front', { note: 'Groove for the bottom, 4 dowel holes, counterbored hole for the knob screw. Print back face down.' }),
      from('drawer-side-r', { note: 'Groove for the bottom, dowel holes at both ends.' }),
      from('drawer-side-l', { note: 'Mirror of the right side.' }),
      from('drawer-back', { note: 'Groove for the bottom, dowel holes at both ends.' }),
      { id: 'drawer-bottom', name: 'Drawer bottom', group: 'Drawer', mat: 'walnut', qty: 1, parent: DR, from: ['drawer-bottom'], explode: byId['drawer-bottom'].explode,
        note: 'Widened 2.8 mm all round to ride in the grooves; slides in from the back before the back goes on.',
        base: () => [K.box(-(DI + 2.8), DI + 2.8, 19, 25, -94.3 - 2.8, CZ1 + 2.8)] },
      from('follower', { note: same }),
      from('divider', { note: same }),
      { id: 'knob', name: 'Drawer knob', group: 'Drawer', mat: 'gilt', qty: 1, parent: DR, from: ['knob'], pos: byId.knob.pos, rot: byId.knob.rot, explode: byId.knob.explode, print: 'axis',
        note: '2.7 mm pilot for an M3 screw from inside the drawer. Print rosette down.',
        base: () => [K.tubeSolid(K.knobRings(0, 10).map(r => ({ y: r.y, ro: r.r, ri: 1.35 })), 144), K.radialSolid(K.knobRings(10), 144)] },
      // printed fasteners
      { id: 'dowel', name: 'Dowel pin Ø4.9 × 15.6', group: 'Fasteners', mat: 'bone', parent: null, print: 'axis', fastener: 'dowel',
        note: 'Print upright in PLA or PETG, or buy 5 mm beech dowels and cut them to 15.5 mm.',
        base: () => [K.radialSolid([{ y: 0, r: 1.95 }, { y: 0.5, r: 2.45 }, { y: 15.1, r: 2.45 }, { y: 15.6, r: 1.95 }], 24)] },
      { id: 'pin-short', name: 'Pin Ø2.9 × 5.9', group: 'Fasteners', mat: 'bone', parent: null, print: 'axis', fastener: 'pin5.9',
        note: 'Frame to end wall. Or cut from 3 mm brass rod.',
        base: () => [K.radialSolid([{ y: 0, r: 1.15 }, { y: 0.3, r: 1.45 }, { y: 5.6, r: 1.45 }, { y: 5.9, r: 1.15 }], 20)] },
      { id: 'pin-long', name: 'Pin Ø2.9 × 9.4', group: 'Fasteners', mat: 'bone', parent: null, print: 'axis', fastener: 'pin9.4',
        note: 'Baffle to roof. Or cut from 3 mm brass rod.',
        base: () => [K.radialSolid([{ y: 0, r: 1.15 }, { y: 0.3, r: 1.45 }, { y: 9.1, r: 1.45 }, { y: 9.4, r: 1.15 }], 20)] }
    );
    kit.filter(k => k.extraCuts).forEach(k => (cuts[k.id] = cuts[k.id] || []).push(...(cuts[k.extraCuts] || [])));
    // cuts that name the column foot land on the column's first shell
    (cuts['column-foot'] || []).forEach(c => cut('column', c.frame, c.make));

    // counts for printed fasteners
    const dowels = fast.filter(f => f.kind === 'dowel').length;
    const pins = fast.filter(f => f.kind === 'pin');
    kit.find(k => k.id === 'dowel').qty = dowels;
    kit.find(k => k.id === 'pin-short').qty = pins.filter(f => f.a1 - f.a0 < 7).length;
    kit.find(k => k.id === 'pin-long').qty = pins.filter(f => f.a1 - f.a0 >= 7).length;

    // build: union shells, subtract cutters (moved into the part's own frame), split if needed
    function make(k) {
      let shells = k.base();
      const list = cuts[k.cutsOf || k.id] || [];
      const Minv = placeM(k).invert();
      if (list.length || k.split) {
        const target = k.cutShell !== undefined ? k.cutShell : null;
        let polys;
        if (target === null) {
          polys = toPolys(shells[0]);
          for (let i = 1; i < shells.length; i++) polys = csgUnion(polys, toPolys(shells[i]));
          shells = [];
        } else { polys = toPolys(shells[target]); shells = shells.filter((_, i) => i !== target); }
        // each cutter only meets the polygons near it: the rest stay out of the BSP entirely,
        // which keeps the planes of one small hole from slicing the whole part
        for (const c of list) {
          const sh = K.transformShell(c.make(), Minv), bb = bbox(sh);
          const g = [[bb[0][0] - 1, bb[0][1] - 1, bb[0][2] - 1], [bb[1][0] + 1, bb[1][1] + 1, bb[1][2] + 1]];
          const near = [], far = [];
          for (const p of polys) (overlap(polyBox(p), g) ? near : far).push(p);
          polys = far.concat(csgSub(near, toPolys(sh)));
        }
        if (k.split) {
          const big = 400, q = [];
          const xs = [[-big, 0], [0, big]], zs = k.split === 'quad' ? [[-big, 0], [0, big]] : [[-big, big]];
          for (const [x0, x1] of xs) for (const [z0, z1] of zs) q.push(toShell(csgAnd(polys, toPolys(K.box(x0, x1, -big, big * 3, z0, z1)))));
          return { pieces: q, rest: shells };
        }
        shells.unshift(toShell(polys));
      }
      return { pieces: null, rest: shells };
    }
    const out = [];
    for (let i = 0; i < kit.length; i++) {
      const k = kit[i];
      if (onProgress) onProgress(i, kit.length, k);
      await new Promise(res => setTimeout(res, 0));
      const r = make(k);
      if (r.pieces) {
        const order = k.split === 'quad' ? [0, 1, 2, 3] : [0, 1];
        order.forEach((pi, n) => {
          const sgx = pi < (k.split === 'quad' ? 2 : 1) ? -1 : 1, sgz = k.split === 'quad' ? (pi % 2 === 0 ? -1 : 1) : 0;
          const nm = k.split === 'quad' ? (sgz > 0 ? 'front ' : 'back ') + (sgx < 0 ? 'left' : 'right') : (sgx < 0 ? 'left' : 'right') + ' half';
          out.push(Object.assign({}, k, { id: k.id + '-' + (n + 1), name: k.name + ', ' + nm, qty: 1, group: k.group, piece: n,
            from: n === 0 ? k.from : [], nudge: [sgx * 18, 0, sgz * 18], shells: () => [r.pieces[pi], ...r.rest], _shells: [r.pieces[pi], ...r.rest] }));
        });
      } else out.push(Object.assign({}, k, { shells: () => r.rest, _shells: r.rest }));
    }

    // hardware list
    const bom = new Map();
    fast.forEach(f => { if (f.kind !== 'dowel' && f.kind !== 'pin') bom.set(f.label, (bom.get(f.label) || 0) + 1); });
    const hardware = [...bom.entries()].map(([label, n]) => ({ label, n }));
    hardware.push({ label: 'Epoxy or CA glue for ornaments, consoles, septum and socket ring', n: 1 });
    return { parts: out, fasteners: fast, hardware };
  }

  H.buildKit = buildKit;
})(typeof window !== 'undefined' ? window : globalThis);
