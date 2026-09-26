/* Heliotrope stereoscope — parametric part geometry.
 * Units: millimetres. Every part is built as one or more closed shells in its own
 * "print frame" (y up), then placed in the scene with a position/rotation.
 * Works in the browser (global THREE) and in Node (globalThis.THREE).
 */
(function (root) {
  const THREE = root.THREE;
  const PI = Math.PI, TAU = Math.PI * 2;

  // ───────────────────────────── key dimensions ─────────────────────────────
  const T = 12.7;                 // 1/2" stock
  const L = 173.7;                // body length, lens-board face (z=0) to end-wall face
  const HW0 = 85, HW1 = 116;      // outer half-width at eye end / card end
  const FLOOR_TOP = T;            // 12.7
  const WALL_TOP = 110;
  const ROOF_TOP = WALL_TOP + T;  // 122.7
  const CARD_W = 178, CARD_H = 89;
  const CARD_BOTTOM = 15.7;
  const AXIS_Y = CARD_BOTTOM + CARD_H / 2;   // 60.2 optical axis height
  const CARD_Z = 155;             // card plane
  const LENS_Z = 5;               // nominal lens plane (card 150 mm = f away)
  const LENS_X = 38;              // lens centres 76 mm apart
  const HUB_Z = 88;               // bayonet centre along body
  const BASE_TOP = 136;           // top of the drawer base
  const PED_TOP = BASE_TOP + 229;  // 9" gilt stand above the base
  const PLATE_BOT = PED_TOP - 24;
  const CW = 118, CZ0 = -122, CZ1 = 104, DF = 123;   // base carcase half-width, back, front, drawer face
  const PLATE_POLY = [[-135, -80], [-107, -108], [107, -108], [135, -80], [135, 80], [107, 108], [-107, 108], [-135, 80]];
  const LID_FRONT = 34, LID_REAR = 167.5, LID_H = 14;
  const RAIL_Z0 = 168, GROOVE_Z0 = 172, GROOVE_Z1 = 187, RAIL_Z1 = 191;
  const THREAD = { pitch: 2.5, starts: 3 };  // helicoid: 7.5 mm lead per turn
  const LUG_ANGLES = [0, TAU / 3, 2 * TAU / 3];
  const LOCK_TURN = PI / 6;       // 30° bayonet turn

  const hwOut = z => HW0 + (HW1 - HW0) * z / L;
  const dirLen = Math.hypot(HW1 - HW0, L);
  const nOut = [L / dirLen, -(HW1 - HW0) / dirLen];   // right wall outward normal (x,z)
  const inP0 = [HW0 - T * nOut[0], -T * nOut[1]];
  const slope = (HW1 - HW0) / L;
  const hwIn = z => inP0[0] + slope * (z - inP0[1]);

  // ───────────────────────────── shell builder ──────────────────────────────
  class Shell {
    constructor() { this.p = []; }
    tri(a, b, c) { this.p.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]); }
    quad(a, b, c, d) { this.tri(a, b, c); this.tri(a, c, d); }
    volume() {
      const p = this.p; let v = 0;
      for (let i = 0; i < p.length; i += 9) {
        v += p[i] * (p[i + 4] * p[i + 8] - p[i + 5] * p[i + 7])
           - p[i + 1] * (p[i + 3] * p[i + 8] - p[i + 5] * p[i + 6])
           + p[i + 2] * (p[i + 3] * p[i + 7] - p[i + 4] * p[i + 6]);
      }
      return v / 6;
    }
    orient() {
      if (this.volume() < 0) {
        const p = this.p;
        for (let i = 0; i < p.length; i += 9) for (let k = 0; k < 3; k++) {
          const t = p[i + 3 + k]; p[i + 3 + k] = p[i + 6 + k]; p[i + 6 + k] = t;
        }
      }
      return this;
    }
  }
  function shellFromGeo(g) {
    const s = new Shell();
    const ng = g.index ? g.toNonIndexed() : g;
    s.p = Array.from(ng.attributes.position.array);
    return s.orient();
  }
  function transformShell(s, m4) {
    const v = new THREE.Vector3(), p = s.p;
    for (let i = 0; i < p.length; i += 3) {
      v.set(p[i], p[i + 1], p[i + 2]).applyMatrix4(m4);
      p[i] = v.x; p[i + 1] = v.y; p[i + 2] = v.z;
    }
    if (m4.determinant() < 0) { s.orient(); }
    return s;
  }

  // Build a display geometry: creased normals + box-projected UVs.
  function finalize(shells, crease = 38) {
    let n = 0; shells.forEach(s => n += s.p.length);
    const pos = new Float32Array(n);
    let o = 0; shells.forEach(s => { pos.set(s.p, o); o += s.p.length; });
    const triCount = n / 9;
    const fn = new Float32Array(triCount * 3);
    const map = new Map();
    const key = i => Math.round(pos[i] * 1e3) + ',' + Math.round(pos[i + 1] * 1e3) + ',' + Math.round(pos[i + 2] * 1e3);
    const keys = new Array(n / 3);
    for (let t = 0; t < triCount; t++) {
      const i = t * 9;
      const ax = pos[i + 3] - pos[i], ay = pos[i + 4] - pos[i + 1], az = pos[i + 5] - pos[i + 2];
      const bx = pos[i + 6] - pos[i], by = pos[i + 7] - pos[i + 1], bz = pos[i + 8] - pos[i + 2];
      let cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
      fn[t * 3] = cx; fn[t * 3 + 1] = cy; fn[t * 3 + 2] = cz;   // area-weighted
      for (let k = 0; k < 3; k++) {
        const vi = t * 3 + k, kk = key(vi * 3);
        keys[vi] = kk;
        let arr = map.get(kk); if (!arr) { arr = []; map.set(kk, arr); } arr.push(t);
      }
    }
    const nrm = new Float32Array(n), uv = new Float32Array(n / 3 * 2);
    const cosC = Math.cos(crease * PI / 180);
    for (let t = 0; t < triCount; t++) {
      const fx = fn[t * 3], fy = fn[t * 3 + 1], fz = fn[t * 3 + 2];
      const fl = Math.hypot(fx, fy, fz) || 1;
      const ux = fx / fl, uy = fy / fl, uz = fz / fl;
      const ax = Math.abs(ux), ay = Math.abs(uy), az = Math.abs(uz);
      for (let k = 0; k < 3; k++) {
        const vi = t * 3 + k;
        let sx = 0, sy = 0, sz = 0;
        for (const u of map.get(keys[vi])) {
          const gx = fn[u * 3], gy = fn[u * 3 + 1], gz = fn[u * 3 + 2];
          const gl = Math.hypot(gx, gy, gz) || 1;
          if ((gx * ux + gy * uy + gz * uz) / gl >= cosC) { sx += gx; sy += gy; sz += gz; }
        }
        const sl = Math.hypot(sx, sy, sz) || 1;
        nrm[vi * 3] = sx / sl; nrm[vi * 3 + 1] = sy / sl; nrm[vi * 3 + 2] = sz / sl;
        const x = pos[vi * 3], y = pos[vi * 3 + 1], z = pos[vi * 3 + 2];
        let a, b;
        if (ay >= ax && ay >= az) { a = x; b = z; } else if (ax >= az) { a = z; b = y; } else { a = x; b = y; }
        uv[vi * 2] = a / 260; uv[vi * 2 + 1] = b / 260;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.computeBoundingBox(); g.computeBoundingSphere();
    return g;
  }

  // ───────────────────────────── primitive builders ─────────────────────────
  // rings: [{y, r: number | (θ)=>number}] ascending y (duplicates allowed for steps)
  function radialSolid(rings, n = 96) {
    const s = new Shell();
    const R = rings.map(ring => {
      const out = [];
      for (let j = 0; j < n; j++) {
        const th = j / n * TAU;
        const r = typeof ring.r === 'function' ? ring.r(th) : ring.r;
        out.push([r * Math.cos(th), ring.y, r * Math.sin(th)]);
      }
      return out;
    });
    for (let i = 0; i < R.length - 1; i++) for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      const A = R[i][j], B = R[i][j1], C = R[i + 1][j1], D = R[i + 1][j];
      s.tri(A, D, C); s.tri(A, C, B);
    }
    const b = [0, rings[0].y, 0], t = [0, rings[rings.length - 1].y, 0];
    const last = R[R.length - 1];
    for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      s.tri(b, R[0][j], R[0][j1]);
      s.tri(t, last[j1], last[j]);
    }
    return s.orient();
  }
  // rings: [{y, ro, ri}] with ro/ri number or (θ)=>number
  function tubeSolid(rings, n = 96) {
    const s = new Shell();
    const f = (v, th) => typeof v === 'function' ? v(th) : v;
    const O = [], I = [];
    rings.forEach(ring => {
      const o = [], ii = [];
      for (let j = 0; j < n; j++) {
        const th = j / n * TAU, c = Math.cos(th), sn = Math.sin(th);
        const ro = f(ring.ro, th), ri = f(ring.ri, th);
        o.push([ro * c, ring.y, ro * sn]); ii.push([ri * c, ring.y, ri * sn]);
      }
      O.push(o); I.push(ii);
    });
    for (let i = 0; i < rings.length - 1; i++) for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      let A = O[i][j], B = O[i][j1], C = O[i + 1][j1], D = O[i + 1][j];
      s.tri(A, D, C); s.tri(A, C, B);
      A = I[i][j]; B = I[i][j1]; C = I[i + 1][j1]; D = I[i + 1][j];
      s.tri(A, C, D); s.tri(A, B, C);
    }
    const k = rings.length - 1;
    for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      s.tri(O[0][j], O[0][j1], I[0][j1]); s.tri(O[0][j], I[0][j1], I[0][j]);
      s.tri(O[k][j], I[k][j1], O[k][j1]); s.tri(O[k][j], I[k][j], I[k][j1]);
    }
    return s.orient();
  }
  // polygon (x,z) convex; returns miter data
  function polyFrame(poly) {
    const n = poly.length;
    let cx = 0, cz = 0; poly.forEach(p => { cx += p[0] / n; cz += p[1] / n; });
    const en = [];
    for (let i = 0; i < n; i++) {
      const a = poly[i], b = poly[(i + 1) % n];
      let ex = b[0] - a[0], ez = b[1] - a[1]; const l = Math.hypot(ex, ez); ex /= l; ez /= l;
      let nx = ez, nz = -ex;
      const mx = (a[0] + b[0]) / 2 - cx, mz = (a[1] + b[1]) / 2 - cz;
      if (nx * mx + nz * mz < 0) { nx = -nx; nz = -nz; }
      en.push([nx, nz]);
    }
    const mit = [];
    for (let i = 0; i < n; i++) {
      const a = en[(i - 1 + n) % n], b = en[i];
      const d = 1 + a[0] * b[0] + a[1] * b[1];
      mit.push([(a[0] + b[0]) / d, (a[1] + b[1]) / d]);
    }
    return { mit, c: [cx, cz] };
  }
  const offPt = (poly, mit, i, off, y) => [poly[i][0] + mit[i][0] * off, y, poly[i][1] + mit[i][1] * off];
  // closed section [(off, y)] swept around polygon → ring shell
  function sweepRing(poly, section) {
    const { mit } = polyFrame(poly);
    const s = new Shell(), n = poly.length, m = section.length;
    for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) {
      const i1 = (i + 1) % n, k1 = (k + 1) % m;
      const a = offPt(poly, mit, i, section[k][0], section[k][1]);
      const b = offPt(poly, mit, i1, section[k][0], section[k][1]);
      const c = offPt(poly, mit, i1, section[k1][0], section[k1][1]);
      const d = offPt(poly, mit, i, section[k1][0], section[k1][1]);
      s.quad(a, b, c, d);
    }
    return s.orient();
  }
  // open profile [(off, y)] bottom → top, capped → solid slab with moulded edge
  function profileSolid(poly, profile) {
    const { mit, c } = polyFrame(poly);
    const s = new Shell(), n = poly.length, m = profile.length;
    for (let i = 0; i < n; i++) for (let k = 0; k < m - 1; k++) {
      const i1 = (i + 1) % n;
      const a = offPt(poly, mit, i, profile[k][0], profile[k][1]);
      const b = offPt(poly, mit, i1, profile[k][0], profile[k][1]);
      const cc = offPt(poly, mit, i1, profile[k + 1][0], profile[k + 1][1]);
      const d = offPt(poly, mit, i, profile[k + 1][0], profile[k + 1][1]);
      s.quad(a, b, cc, d);
    }
    const y0 = profile[0][1], y1 = profile[m - 1][1];
    const cb = [c[0], y0, c[1]], ct = [c[0], y1, c[1]];
    for (let i = 0; i < n; i++) {
      const i1 = (i + 1) % n;
      s.tri(cb, offPt(poly, mit, i1, profile[0][0], y0), offPt(poly, mit, i, profile[0][0], y0));
      s.tri(ct, offPt(poly, mit, i, profile[m - 1][0], y1), offPt(poly, mit, i1, profile[m - 1][0], y1));
    }
    return s.orient();
  }
  const toShape = (pts, holes = []) => {
    const V = a => a.map(p => new THREE.Vector2(p[0], p[1]));
    let outer = V(pts);
    if (THREE.ShapeUtils.isClockWise(outer)) outer = outer.reverse();
    const sh = new THREE.Shape(outer);
    holes.forEach(h => { let hv = V(h); if (!THREE.ShapeUtils.isClockWise(hv)) hv = hv.reverse(); sh.holes.push(new THREE.Path(hv)); });
    return sh;
  };
  // plan polygon (x,z) extruded up from y0 by h
  function extrudePlan(poly, holes, y0, h, bevel) {
    const flip = p => [p[0], -p[1]];
    const opts = { depth: h, bevelEnabled: false, curveSegments: 48, steps: 1 };
    if (bevel) Object.assign(opts, { depth: h - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelOffset: -bevel, bevelSegments: 3 });
    const g = new THREE.ExtrudeGeometry(toShape(poly.map(flip), holes.map(hh => hh.map(flip))), opts);
    g.rotateX(-PI / 2);
    g.computeBoundingBox(); g.translate(0, y0 - g.boundingBox.min.y, 0);
    return shellFromGeo(g);
  }
  // front-view shape (x,y) extruded along z from z0 to z1
  function extrudeXY(pts, holes, z0, z1, endFn) {
    const g = new THREE.ExtrudeGeometry(toShape(pts, holes), { depth: z1 - z0, bevelEnabled: false, curveSegments: 48 });
    g.translate(0, 0, z0);
    if (endFn) {   // bevel the board ends to meet tapered walls
      const p = g.attributes.position; let mx = 0;
      for (let i = 0; i < p.count; i++) mx = Math.max(mx, Math.abs(p.getX(i)));
      for (let i = 0; i < p.count; i++) {
        const x = p.getX(i);
        if (Math.abs(Math.abs(x) - mx) < 1e-3) p.setX(i, Math.sign(x) * endFn(p.getZ(i)));
      }
    }
    return shellFromGeo(g);
  }
  // side section (z,y) extruded along x from x0 to x1
  function extrudeZY(pts, x0, x1) {
    const g = new THREE.ExtrudeGeometry(toShape(pts.map(p => [-p[0], p[1]])), { depth: x1 - x0, bevelEnabled: false, curveSegments: 24 });
    g.rotateY(PI / 2); g.translate(x0, 0, 0);
    return shellFromGeo(g);
  }
  function box(x0, x1, y0, y1, z0, z1) {
    const s = new Shell();
    const v = (x, y, z) => [x, y, z];
    const P = [v(x0, y0, z0), v(x1, y0, z0), v(x1, y1, z0), v(x0, y1, z0), v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1)];
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]
      .forEach(q => s.quad(P[q[0]], P[q[1]], P[q[2]], P[q[3]]));
    return s.orient();
  }
  const circle = (cx, cy, r, n = 64) => Array.from({ length: n }, (_, i) => [cx + r * Math.cos(i / n * TAU), cy + r * Math.sin(i / n * TAU)]);
  const insetPoly = (poly, d) => { const { mit } = polyFrame(poly); return poly.map((p, i) => [p[0] - mit[i][0] * d, p[1] - mit[i][1] * d]); };

  // Heightfield slab on a rectangle; h(u,v) with u across x, v across z.
  function heightRect(w, d, res, base, hFn) {
    const nx = Math.round(w / res), nz = Math.round(d / res);
    const H = new Float32Array((nx + 1) * (nz + 1));
    for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
      H[j * (nx + 1) + i] = base + hFn(-w / 2 + i * w / nx, -d / 2 + j * d / nz);
    }
    const P = (i, j) => [-w / 2 + i * w / nx, H[j * (nx + 1) + i], -d / 2 + j * d / nz];
    const B = (i, j) => [-w / 2 + i * w / nx, 0, -d / 2 + j * d / nz];
    const s = new Shell();
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const a = P(i, j), b = P(i + 1, j), c = P(i + 1, j + 1), dd = P(i, j + 1);
      s.tri(a, b, c); s.tri(a, c, dd);
    }
    const loop = [];
    for (let i = 0; i < nx; i++) loop.push([i, 0]);
    for (let j = 0; j < nz; j++) loop.push([nx, j]);
    for (let i = nx; i > 0; i--) loop.push([i, nz]);
    for (let j = nz; j > 0; j--) loop.push([0, j]);
    const cb = [0, 0, 0];
    for (let k = 0; k < loop.length; k++) {
      const a = loop[k], b = loop[(k + 1) % loop.length];
      const at = P(a[0], a[1]), bt = P(b[0], b[1]), ab = B(a[0], a[1]), bb = B(b[0], b[1]);
      s.tri(bt, at, ab); s.tri(bt, ab, bb);
      s.tri(cb, bb, ab);
    }
    return s.orient();
  }
  // Heightfield on an ellipse (semi-axes a along x, b along z); h(X,Z,ρ,θ)
  function heightOval(a, b, nR, nT, base, hFn) {
    const s = new Shell();
    const P = (i, j) => {
      const rho = i / nR, th = j / nT * TAU;
      const x = a * rho * Math.cos(th), z = b * rho * Math.sin(th);
      return [x, base + hFn(x, z, rho, th), z];
    };
    const C = [0, base + hFn(0, 0, 0, 0), 0];
    for (let j = 0; j < nT; j++) {
      const j1 = (j + 1) % nT;
      s.tri(C, P(1, j1), P(1, j));
      for (let i = 1; i < nR; i++) {
        const A = P(i, j), B = P(i, j1), Cc = P(i + 1, j1), D = P(i + 1, j);
        s.tri(A, B, Cc); s.tri(A, Cc, D);
      }
      const at = P(nR, j), bt = P(nR, j1);
      const ab = [at[0], 0, at[2]], bb = [bt[0], 0, bt[2]];
      s.tri(at, bt, bb); s.tri(at, bb, ab);
      s.tri([0, 0, 0], ab, bb);
    }
    return s.orient();
  }

  // ───────────────────────────── ornament fields ────────────────────────────
  const G = (d2, s) => Math.exp(-d2 / (s * s));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const hash = (x, y) => { const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); };
  const tube = (d, w, h) => d < w ? h * Math.sqrt(1 - (d / w) * (d / w)) : 0;
  function polyDist(pts, x, y) {
    let best = 1e9, bt = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const ex = b[0] - a[0], ey = b[1] - a[1];
      const t = clamp(((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey || 1), 0, 1);
      const d = Math.hypot(x - a[0] - ex * t, y - a[1] - ey * t);
      if (d < best) { best = d; bt = (i + t) / (pts.length - 1); }
    }
    return [best, bt];
  }
  // corner C-scroll (spiral) used by the Sun panel
  const scrollPts = (() => {
    const pts = [];
    for (let t = 0; t <= 3.2 * PI; t += 0.08) {
      const r = 8.5 * Math.exp(-0.15 * t), a = t + 0.4 * PI;
      pts.push([57 + r * Math.cos(a), 32 + r * Math.sin(a)]);
    }
    return pts;
  })();
  function leaf(u, v, bx, by, dx, dy, len, wid, h) {
    const l = Math.hypot(dx, dy); dx /= l; dy /= l;
    const px = u - bx, py = v - by;
    const t = (px * dx + py * dy) / len, q = -px * dy + py * dx;
    if (t < 0 || t > 1) return 0;
    const bend = 0.18 * wid * Math.sin(t * PI);
    const w = wid * Math.sin(Math.pow(t, 0.7) * PI) * (1 - 0.25 * t);
    const qq = q - bend;
    if (w < 1e-4 || Math.abs(qq) > w) return 0;
    const lobes = 0.8 + 0.2 * Math.cos(t * 5 * TAU);
    const body = h * Math.sqrt(1 - (qq / w) * (qq / w)) * (1 - 0.35 * t) * lobes;
    const rib = -0.35 * h * Math.exp(-(qq * qq) / 0.25);
    return Math.max(0, body + rib);
  }

  // The Sun panel (lid): 150 × 100, face "up" is +v (toward the card end when closed).
  function sunHeight(u, v) {
    const au = Math.abs(u), av = Math.abs(v);
    let h = 0;
    // border: rounded bead, pearl row, fillet
    const d = Math.min(75 - au, 50 - av);
    if (d < 4.5) h = Math.max(h, 3.8 * Math.sqrt(Math.max(0, 1 - ((d - 2.25) / 2.25) ** 2)) + 0.2);
    const s = (75 - au) < (50 - av) ? v : u;
    const pr = Math.hypot(d - 7, ((s % 3.4) + 3.4) % 3.4 - 1.7);
    h = Math.max(h, tube(pr, 1.25, 2.1));
    h = Math.max(h, tube(Math.abs(d - 9.6), 0.55, 0.9));
    // sun
    const r = Math.hypot(u, v), th = Math.atan2(v, u);
    const R0 = 19;
    if (r < R0) {
      let f = 3.0 + 3.0 * Math.sqrt(1 - (r / R0) ** 2);
      for (const sx of [-1, 1]) {
        const ex = u - sx * 6.3, ey = v - 3.6;
        f += -1.5 * G(ex * ex + ey * ey, 3.1) + 1.2 * G(ex * ex + ey * ey, 1.7);
        const lid = ey - 1.6 + 0.05 * ex * ex;
        f += 0.55 * Math.exp(-lid * lid / 0.5) * G(ex * ex, 3.6);
        const bx = u - sx * 6.6, by = v - 8.2 + 0.035 * bx * bx;
        f += 0.9 * Math.exp(-by * by / 1.1) * G(bx * bx, 4.6);
        f += 0.85 * G((u - sx * 9.5) ** 2 + (v + 3.5) ** 2, 4.6);
        f += -0.55 * G((u - sx * 2.3) ** 2 + (v + 5.9) ** 2, 1.0);
      }
      f += 1.5 * Math.exp(-u * u / 4.4) * smooth(-5.5, -3, v) * (1 - smooth(4, 6.5, v));
      f += 1.25 * G(u * u + (v + 4.7) ** 2, 2.5);
      const ul = v + 9.4 - 0.035 * u * u, ll = v + 12.1 - 0.02 * u * u, ms = v + 10.7 - 0.03 * u * u;
      f += 0.95 * Math.exp(-((u / 5.4) ** 4)) * Math.exp(-ul * ul / 1.0);
      f += 1.15 * Math.exp(-((u / 4.4) ** 4)) * Math.exp(-ll * ll / 1.5);
      f += -0.75 * Math.exp(-((u / 5.6) ** 4)) * Math.exp(-ms * ms / 0.2);
      f += 0.7 * G(u * u + (v + 15.5) ** 2, 3.6);
      f *= smooth(R0, R0 - 1.2, r) * 0.25 + 0.75;
      h = Math.max(h, f);
    }
    // flame mane
    if (r >= R0 - 1 && r < 27) {
      const wav = 0.72 + 0.28 * Math.cos(18 * th + 2.4 * Math.sin(r * 0.9));
      h = Math.max(h, 4.4 * wav * Math.sqrt(Math.max(0, 1 - ((r - 22.5) / 4.6) ** 2)));
    }
    // rays: 16 straight, 16 flaming
    if (r > 25) {
      const lim = Math.min(64 / Math.max(Math.abs(Math.cos(th)), 1e-6), 38.5 / Math.max(Math.abs(Math.sin(th)), 1e-6), 58);
      const fu = Math.abs(Math.atan2(av, au));
      const cornerCut = 1 - 0.3 * Math.exp(-(((fu - 0.52) / 0.2) ** 2));
      const step = TAU / 32;
      const k = Math.round(th / step);
      for (const kk of [k - 1, k, k + 1]) {
        const flame = ((kk % 2) + 2) % 2 === 1;
        let dth = th - kk * step;
        if (flame) dth -= 0.085 * Math.sin(0.38 * r) * (r - 25) / 30;
        const Lr = (flame ? 0.74 : 1) * lim * cornerCut;
        if (r > Lr) continue;
        const t = (r - 25) / (Lr - 25);
        const w = (flame ? 3.1 : 4.3) * (1 - t) + 0.2;
        const lat = Math.abs(dth) * r;
        if (lat > w) continue;
        let hh = (flame ? 2.9 : 3.7) * Math.sqrt(1 - t * 0.85) * Math.sqrt(1 - (lat / w) ** 2);
        if (!flame) hh += 0.45 * Math.exp(-lat * lat / 0.35) * (1 - t);
        h = Math.max(h, hh);
      }
    }
    // corner scrolls with acanthus
    if (au > 38 && av > 16) {
      const [dd, tt] = polyDist(scrollPts, au, av);
      h = Math.max(h, tube(dd, 2.2 - 1.1 * tt, 3.1 - 0.9 * tt));
      h = Math.max(h, 2.2 * G((au - 57) ** 2 + (av - 32) ** 2, 1.4));
      h = Math.max(h, leaf(au, av, 50, 38, -1, 0.02, 16, 3.6, 2.4));
      h = Math.max(h, leaf(au, av, 63, 25, -0.08, -1, 13, 3.2, 2.3));
      h = Math.max(h, leaf(au, av, 49, 29, -0.9, -0.55, 10, 2.6, 1.9));
    }
    // side palmettes
    if (au > 52) {
      for (let k = -2; k <= 2; k++) {
        const a = PI + k * 0.42;
        h = Math.max(h, leaf(au, v, 64.5, 0, Math.cos(a), Math.sin(a), 9.5 - Math.abs(k) * 1.3, 1.9, 2.3));
      }
      h = Math.max(h, 2.4 * G((au - 64.5) ** 2 + v * v, 2.0));
    }
    // punched ground (pointillé)
    if (h < 0.35 && d > 10) {
      const gx = u / 1.3, gy = v / 1.3;
      const cx = Math.floor(gx), cy = Math.floor(gy);
      const jx = cx + 0.5 + (hash(cx, cy) - 0.5) * 0.5, jy = cy + 0.5 + (hash(cy, cx) - 0.5) * 0.5;
      h = Math.max(h, 0.22 * G((gx - jx) ** 2 + (gy - jy) ** 2, 0.28));
    }
    return h;
  }

  // Laurel wreath medallion: oval 80 × 60 (X along a, Z along b)
  function wreathHeight(X, Z, rho, th) {
    let h = 0;
    // rim bead
    h = Math.max(h, tube(Math.abs(rho - 0.94) * 34, 1.7, 2.2) + (rho > 0.88 ? 0.3 : 0));
    const ang = Math.atan2(Z / 30, X / 40);
    // wreath: leaves in pairs along ellipse ring rho≈0.7
    const nL = 22;
    for (let side = -1; side <= 1; side += 2) {
      for (let k = 0; k < nL; k++) {
        if (k === 0) continue;
        const a = -PI / 2 + side * (k / nL) * PI * 0.94;
        const px = 40 * 0.7 * Math.cos(a), pz = 30 * 0.7 * Math.sin(a);
        if (Math.abs(X - px) > 8 || Math.abs(Z - pz) > 8) continue;
        const tx = -40 * Math.sin(a) * side, tz = 30 * Math.cos(a) * side;
        const tl = Math.hypot(tx, tz);
        for (const o of [-1, 1]) {
          const nx = tz / tl * o, nz = -tx / tl * o;
          h = Math.max(h, leaf(X, Z, px, pz, tx / tl * 0.7 + nx * 0.7, tz / tl * 0.7 + nz * 0.7, 7.2, 1.8, 2.4));
        }
      }
    }
    // stem
    const ring = Math.abs(Math.hypot(X / 40, Z / 30) - 0.7) * 34;
    if (Math.sin(ang) < 0.95) h = Math.max(h, tube(ring, 0.7, 1.4));
    // central star-sun
    const r = Math.hypot(X, Z), t8 = Math.atan2(Z, X);
    if (r < 17) {
      const star = Math.pow(Math.abs(Math.cos(8 * t8 / 2)), 3);
      const ray = (1 - r / 17) * (1.4 + 2.2 * star);
      h = Math.max(h, ray);
      h = Math.max(h, 3.6 * Math.sqrt(Math.max(0, 1 - (r / 6.5) ** 2)) + 0.6);
    }
    // ribbon bow at bottom
    h = Math.max(h, 2.2 * G((X - 3.5) ** 2 + (Z + 21) ** 2, 2.6), 2.2 * G((X + 3.5) ** 2 + (Z + 21) ** 2, 2.6));
    if (h < 0.3) {
      const gx = X / 1.3, gy = Z / 1.3, cx = Math.floor(gx), cy = Math.floor(gy);
      h = Math.max(h, 0.2 * G((gx - cx - 0.5) ** 2 + (gy - cy - 0.5) ** 2, 0.28));
    }
    return h;
  }

  // console scroll outline (r, y)
  function consoleShape(dx = -7) {
    const ctrl = [[80, 147], [85, 150.5], [86, 145], [81, 140], [73, 141.5], [69.5, 149], [72.5, 157],
      [72, 168], [63, 180], [51, 192], [42, 203], [37, 212], [35.5, 219], [39, 224.5], [44.5, 223], [45, 217.5], [41, 216]];
    const curve = new THREE.CatmullRomCurve3(ctrl.map(p => new THREE.Vector3(p[0] + dx, p[1], 0)), false, 'centripetal');
    const N = 120, pts = curve.getSpacedPoints(N);
    const width = t => {
      if (t < 0.28) return 2.2 + 11 * smooth(0, 0.28, t);
      if (t < 0.62) return 13.2 - 3.5 * smooth(0.28, 0.62, t);
      return 9.7 - 7.7 * smooth(0.62, 1, t);
    };
    const L1 = [], L2 = [];
    for (let i = 0; i <= N; i++) {
      const p = pts[i], q = pts[Math.min(N, i + 1)], o = pts[Math.max(0, i - 1)];
      let tx = q.x - o.x, ty = q.y - o.y; const tl = Math.hypot(tx, ty); tx /= tl; ty /= tl;
      const w = width(i / N) / 2;
      L1.push([p.x - ty * w, p.y + tx * w]); L2.push([p.x + ty * w, p.y - tx * w]);
    }
    return L1.concat(L2.reverse());
  }

  // cresting for the lid rest: front-view silhouette (x, y) above y0
  function crestingOutline(half, y0) {
    const pts = [];
    const N = 80;
    for (let i = 0; i <= N; i++) {
      const x = -half + 2 * half * i / N, ax = Math.abs(x) / half;
      let y = 12 + 22 * Math.pow(Math.cos(ax * PI / 2), 1.6);
      y += 2.2 * Math.max(0, Math.cos(ax * 7 * PI)) * (1 - ax * 0.4);   // scalloped crest
      if (ax > 0.9) y = 12 + (1 - ax) * 30;
      pts.push([x, y0 + y]);
    }
    pts.push([half, y0], [-half, y0]);
    return pts.reverse();
  }

  // column profile rings (y, r(θ)); every flare is ≤45° so it prints upright without supports
  function columnRings(y0 = -1e9, y1 = 1e9) {
    const B = BASE_TOP, C = PLATE_BOT;
    const prof = [
      [B, 54], [B + 3, 56], [B + 6, 55], [B + 9, 50], [B + 11, 44], [B + 14, 36], [B + 18, 33], [B + 26, 38], [B + 36, 45], [B + 44, 47],
      [B + 52, 44], [B + 59, 38], [B + 64, 33], [B + 67, 36], [B + 70, 37], [B + 73, 33], [B + 77, 27], [B + 82, 24.5],
      [C - 51, 24.5], [C - 47, 29], [C - 44, 31], [C - 41, 29], [C - 37, 27], [C - 29, 30], [C - 21, 35], [C - 15, 40],
      [C - 9, 46], [C - 4, 51], [C, 55]];
    const sh0 = B + 82, sh1 = C - 51, bell0 = C - 37, bell1 = C - 11;
    const rings = [];
    const rAt = y => { for (let i = 0; i < prof.length - 1; i++) if (y <= prof[i + 1][0]) { const t = (y - prof[i][0]) / (prof[i + 1][0] - prof[i][0]); return prof[i][1] + (prof[i + 1][1] - prof[i][1]) * (t * t * (3 - 2 * t)); } return prof[prof.length - 1][1]; };
    const ringAt = y => ({ y, r: th => {
      let rr = rAt(y);
      if (y > B + 18 && y < B + 58) rr += 2.8 * Math.sin((y - B - 18) / 40 * PI) * (Math.pow(Math.abs(Math.cos(8 * th)), 0.6) - 0.35);
      if (y > sh0 && y < sh1) {
        const e = smooth(sh0, sh0 + 6, y) * (1 - smooth(sh1 - 6, sh1, y));
        rr += 4.2 * e * Math.pow(0.5 + 0.5 * Math.cos(4 * th - (y - sh0) * 0.085), 1.6);
      }
      if (y > bell0 && y < bell1) {
        const t2 = (y - bell0) / (bell1 - bell0), a = ((th * 8 / TAU) % 1 + 1) % 1, dd = Math.abs(a - 0.5) * 2;
        rr += 3.6 * Math.max(0, t2 * 1.1 - dd * (0.4 + (1 - t2)));
      }
      if (y > C - 10 && y < C - 1) rr += 1.0 * Math.abs(Math.cos(12 * th)) * Math.sin((y - C + 10) / 9 * PI);
      return rr;
    } });
    const lo = Math.max(B, y0), hi = Math.min(C, y1);
    const n = Math.ceil((hi - lo) / 0.8);
    for (let k = 0; k <= n; k++) rings.push(ringAt(lo + (hi - lo) * k / n));
    return rings;
  }
  const KNOB_PROF = [[0, 17], [1.5, 18], [3, 16], [3.5, 7], [6, 6], [8, 7.5], [10, 9.5], [11, 10.5], [15, 12], [19, 11], [22, 8], [23.5, 5], [24.5, 2.5], [25.5, 0.01]];
  function knobRings(y0 = -1, y1 = 99) {
    const rings = [];
    KNOB_PROF.forEach(([y, r], i) => {
      if (y < y0 - 1e-9 || y > y1 + 1e-9) return;
      if (y <= 3) rings.push({ y, r: th => r + (i < 3 ? 2.6 * (Math.pow(Math.abs(Math.cos(6 * th)), 0.5) - 0.3) : 0) });
      else if (y > 9 && y < 21) rings.push({ y, r: th => r * (1 + 0.07 * Math.abs(Math.cos(6 * th))) });
      else rings.push({ y, r });
    });
    return rings;
  }

  // ───────────────────────────── part list ─────────────────────────────────
  function buildParts(opts = {}) {
    const q = opts.quality || 1;
    const parts = [];
    const add = (p) => { parts.push(p); return p; };
    const bodyPoly = [[-HW0, 0], [HW0, 0], [HW1, L], [-HW1, L]];

    // ── Viewer body (walnut) ──
    const floorSection = [[-8, 0], [10, 0], [10, 3], [9.5, 4.5], [8, 5.5], [6.5, 6.5], [5.5, 8], [5, 9.5], [4, 11], [2.5, 12], [1, 12.7], [-8, 12.7]];
    add({ id: 'floor', name: 'Floor board', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: '½" walnut. Ogee edge routed; 124 mm hole takes the bayonet socket.',
      shells: () => [sweepRing(bodyPoly, floorSection),
        extrudePlan(insetPoly(bodyPoly, 7.5), [circle(0, HUB_Z, 62, 96)], 0, T)],
      explode: [0, -70, 0] });
    const wallR = [[HW0, 0], [HW1, L], [hwIn(L), L], [hwIn(0), 0]];
    add({ id: 'wall-r', name: 'Side wall, right', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: '½" walnut. Ends bevelled 10.1° to meet the tapered plan.', down: [nOut[0], 0, nOut[1]],
      shells: () => [extrudePlan(wallR, [], FLOOR_TOP, WALL_TOP - FLOOR_TOP)], explode: [95, 0, 0] });
    add({ id: 'wall-l', name: 'Side wall, left', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: 'Mirror of the right wall.', down: [-nOut[0], 0, nOut[1]],
      shells: () => [extrudePlan(wallR.map(p => [-p[0], p[1]]), [], FLOOR_TOP, WALL_TOP - FLOOR_TOP)], explode: [-95, 0, 0] });
    const boardH = [FLOOR_TOP, WALL_TOP];
    add({ id: 'lens-board', name: 'Lens board', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: '½" walnut. Two 50.2 mm bores on 76 mm centres at axis height 60.2 mm.', down: [0, 0, -1],
      shells: () => {
        const w = hwIn(T);
        return [extrudeXY([[-w, boardH[0]], [w, boardH[0]], [w, boardH[1]], [-w, boardH[1]]],
          [circle(-LENS_X, AXIS_Y, 25.1, 72), circle(LENS_X, AXIS_Y, 25.1, 72)], 0, T, hwIn)];
      }, explode: [0, 0, -80] });
    const ENDZ0 = L - T;
    add({ id: 'end-wall', name: 'End wall', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: '½" walnut with a 156 × 61 mm window for the ground glass.', down: [0, 0, 1],
      shells: () => {
        const w = hwIn(L);
        return [extrudeXY([[-w, boardH[0]], [w, boardH[0]], [w, boardH[1]], [-w, boardH[1]]],
          [[[-78, 30], [78, 30], [78, 91], [-78, 91]]], ENDZ0, L, hwIn)];
      }, explode: [0, 0, 95] });
    const sky = [[-(hwIn(40) - 4), 40], [hwIn(40) - 4, 40], [hwIn(145) - 4, 145], [-(hwIn(145) - 4), 145]];
    const slot = [[-91, 153], [91, 153], [91, 157], [-91, 157]];
    const roofPoly = [[-(hwOut(-6) + 6.1), -6], [hwOut(-6) + 6.1, -6], [hwOut(RAIL_Z1) + 6.1, RAIL_Z1], [-(hwOut(RAIL_Z1) + 6.1), RAIL_Z1]];
    add({ id: 'roof', name: 'Roof with skylight', group: 'Viewer body', mat: 'walnut', qty: 1, parent: 'viewer',
      note: '½" walnut, two laminations: the lower one forms a 4 mm ledge that the skylight glass drops onto. Card slot 182 × 4 mm.',
      shells: () => [
        extrudePlan(roofPoly, [sky, slot], WALL_TOP + 3, T - 3, 1.5),
        extrudePlan(roofPoly, [insetPoly(sky, 4), slot], WALL_TOP, 3)],
      explode: [0, 100, 0] });
    const cornice = [[-1, 101], [0.5, 101], [1.8, 102.4], [2.5, 104.2], [3.3, 105.8], [4.6, 107.1], [5.7, 108.2], [6.1, 110], [-1, 110]];
    add({ id: 'cornice', name: 'Cornice moulding', group: 'Ornament', mat: 'gilt', qty: 1, parent: 'viewer', print: true,
      note: 'Gilt cove wrapping the four walls under the roof. Print in 4 mitred runs if your bed is small.',
      shells: () => [sweepRing(bodyPoly, cornice)], explode: [0, 55, 0] });

    // ── Light path: gate, glass, baffles ──
    const GW = 98, GB = FLOOR_TOP;
    add({ id: 'gate', name: 'Card gate', group: 'Optics', mat: 'black', qty: 1, parent: 'viewer', print: true,
      note: 'Matte black. Front mask, 4 mm card channel and ground-glass seat in one print. The card stands on its 3 mm sill.',
      shells: () => [
        extrudeXY([[-GW, GB], [GW, GB], [GW, 110], [86, 110], [86, 22], [-86, 22], [-86, 110], [-GW, 110]], [], 150, 153),
        extrudeXY([[-GW, GB], [GW, GB], [GW, 110], [90.5, 110], [90.5, CARD_BOTTOM], [-90.5, CARD_BOTTOM], [-90.5, 110], [-GW, 110]], [], 152.99, 157.01),
        extrudeXY([[-GW, GB], [GW, GB], [GW, 110], [-GW, 110]], [[[-86, 22], [86, 22], [86, 100], [-86, 100]]], 157, 159)],
      explode: [0, 0, 40] });
    add({ id: 'ground-glass', name: 'Ground glass, end', group: 'Optics', mat: 'groundglass', qty: 1, parent: 'viewer', buy: true,
      note: '194 × 90 × 2 mm glass, ground side toward the lenses. Cut, not printed.',
      shells: () => [box(-97, 97, 16, 106, 159, 161)], explode: [0, 0, 62] });
    add({ id: 'sky-glass', name: 'Skylight glass', group: 'Optics', mat: 'groundglass', qty: 1, parent: 'viewer', buy: true,
      note: 'Ground glass 2.5 mm, trapezoid (template in STL). Drops onto the roof ledge from above.',
      shells: () => [extrudePlan(insetPoly(sky, 0.6), [], 113, 2.5)], explode: [0, 150, 0] });
    add({ id: 'septum', name: 'Septum', group: 'Optics', mat: 'black', qty: 1, parent: 'viewer', print: true,
      note: 'Keeps each eye on its own picture. Matte black, 3 mm.',
      shells: () => [box(-1.5, 1.5, 14.5, 92, T, 120)], explode: [0, 60, -20] });
    add({ id: 'baffle', name: 'Glare baffle', group: 'Optics', mat: 'black', qty: 1, parent: 'viewer', print: true,
      note: 'Hangs from the roof ahead of the skylight so no sky shines into the lenses.',
      shells: () => [box(-76, 76, 86, 110, 26, 29)], explode: [0, 70, -40] });

    // ── Eyepieces ──
    const tri = f => 1 - Math.abs(2 * (f - Math.floor(f)) - 1);
    const lead = THREAD.pitch * THREAD.starts;
    const femaleR = zv => th => 22.0 + 1.3 * tri(zv / THREAD.pitch - THREAD.starts * th / TAU);
    const maleR = zl => th => 21.6 + 1.3 * tri((zl + LENS_Z) / THREAD.pitch - THREAD.starts * th / TAU);
    const petals = (th, n, a) => a * (Math.pow(Math.abs(Math.cos(n * th / 2)), 0.6) - 0.5);
    add({ id: 'sleeve', name: 'Eyepiece sleeve', group: 'Eyepieces', mat: 'gilt', qty: 2, parent: 'viewer', print: 'axisDown', pair: true,
      note: 'Fixed in the lens board. Rosette flange; 3-start internal helicoid (2.5 mm pitch, 7.5 mm per turn).',
      shells: () => {
        const rings = [];
        const zs = [];
        for (let z = 18; z >= 0.001; z -= 0.3 / q) zs.push(z);
        zs.push(0, 0);
        for (let z = -0.3; z >= -7; z -= 0.3 / q) zs.push(z);
        zs.push(-7);
        let flangeStarted = false;
        zs.forEach((zv, idx) => {
          let ro;
          if (zv > 0 || (zv === 0 && !flangeStarted)) { ro = 25; if (zv === 0) flangeStarted = true; }
          else {
            const t = -zv / 7;
            const prof = 27 + 7 * (1 - t);   // 45° cone: prints outer face down without support
            ro = th => prof + petals(th, 12, 2.6 * (1 - t * 0.5));
          }
          rings.push({ y: -zv, ro, ri: femaleR(zv) });
        });
        return [tubeSolid(rings, Math.round(144 * q))];
      }, explode: [0, 0, -100] });
    add({ id: 'draw-tube', name: 'Eyepiece draw tube & eyecup', group: 'Eyepieces', mat: 'gilt', qty: 2, parent: 'viewer', print: 'axisDown', pair: true,
      note: 'Screws into the sleeve to focus (−8 to +15 mm travel). Knurled grip, gadrooned eyecup with pearl rim. Holds one 38 mm f150 lens.',
      shells: () => {
        const rings = [];
        const push = (zl, ro, ri) => rings.push({ y: -zl, ro, ri });
        const ri = zl => zl < -40 ? 19.2 + 3.8 * ((-40 - zl) / 5) ** 1.5 : zl < 2.2 ? 19.2 : zl < 4.2 ? 17.5 : 20;
        const zsteps = [];
        for (let z = 15; z > -10; z -= 0.25 / q) zsteps.push(z);
        zsteps.push(-10);
        for (let z = -10.4; z > -27; z -= 1.5) zsteps.push(z);
        zsteps.push(-27);
        for (let z = -27.4; z > -37; z -= 0.4) zsteps.push(z);
        zsteps.push(-37);
        for (let z = -37.4; z > -45; z -= 0.3 / q) zsteps.push(z);
        zsteps.push(-45);
        // insert duplicate rings where the bore steps
        const all = [];
        zsteps.forEach(z => all.push(z));
        all.sort((a, b) => b - a);
        const stepZ = [4.2, 2.2];
        const zsFinal = [];
        all.forEach(z => { zsFinal.push(z); });
        stepZ.forEach(sz => { const i = zsFinal.findIndex(z => z < sz); zsFinal.splice(i, 0, sz + 1e-4, sz - 1e-4); });
        zsFinal.forEach(zl => {
          let ro;
          if (zl >= -10) ro = maleR(zl);
          else if (zl > -27) ro = 21.4;
          else if (zl > -37) {
            const e = Math.min(zl + 37, -27 - zl);
            const base = 25 + Math.min(2, zl + 37, (-27 - zl) * 2);   // 45° step up from the eyecup when printed eyecup-down
            ro = th => base + 0.55 * tri(30 * th / TAU) * Math.min(1, e);
          } else {
            const t = (-37 - zl) / 8;
            const flare = 25 + 3.2 * t * t;
            const bead = t > 0.62 ? 1.6 * Math.sqrt(Math.max(0, 1 - ((t - 0.81) / 0.19) ** 2)) : 0;
            ro = th => flare + bead * (0.55 + 0.45 * Math.abs(Math.cos(9 * th))) + (t < 0.6 ? 0.9 * Math.abs(Math.cos(8 * th)) * Math.sin(t / 0.6 * PI) : 0);
          }
          push(zl, ro, ri(zl));
        });
        return [tubeSolid(rings, Math.round(144 * q))];
      }, explode: [0, 0, -150] });
    add({ id: 'lens', name: 'Lens, 38 mm f150 biconvex', group: 'Eyepieces', mat: 'lens', qty: 2, parent: 'viewer', buy: true, noExport: true, pair: true,
      note: 'Your existing lenses. Seat against the lip inside the draw tube, flat edge toward the eye.',
      shells: () => {
        const R = 150, rings = [];
        const rr = y => Math.abs(y) <= 1 ? 19 : Math.sqrt(Math.max(0, R * R - (R - (2.2 - Math.abs(y))) ** 2));
        for (let i = 0; i <= 40; i++) { const y = -2.2 + 4.4 * i / 40; rings.push({ y, r: rr(y) }); }
        return [radialSolid(rings, 72)];
      }, explode: [0, 0, -195] });
    add({ id: 'lens-ring', name: 'Lens retaining ring', group: 'Eyepieces', mat: 'black', qty: 2, parent: 'viewer', print: 'axis', pair: true,
      note: 'Press fit behind the lens (a drop of CA holds it).',
      shells: () => [tubeSolid([{ y: 2.2, ro: 19.1, ri: 16.5 }, { y: 5.2, ro: 19.1, ri: 16.5 }], 96)], explode: [0, 0, -225] });

    // ── Bayonet ──
    const lugHalf = 8.8 * PI / 180, notchHalf = 11 * PI / 180;
    const angDiff = (a, b) => { let d = (a - b) % TAU; if (d > PI) d -= TAU; if (d < -PI) d += TAU; return d; };
    const inNotch = th => LUG_ANGLES.some(a => Math.abs(angDiff(th, a + PI + LOCK_TURN)) < notchHalf);
    const inGallery = th => LUG_ANGLES.some(a => { const d = angDiff(th, a + PI); return d > -notchHalf && d < LOCK_TURN + notchHalf; });
    add({ id: 'socket', name: 'Bayonet socket (viewer)', group: 'Mount', mat: 'brass', qty: 1, parent: 'viewer', print: 'axis',
      note: 'Glued into the floor, flush underneath. Three notches, a 30° gallery with end stops. Bed three 6 × 3 mm magnets in the ceiling.',
      shells: () => {
        const lip = th => inNotch(th) ? 53.5 : 46, gal = th => inGallery(th) ? 53.5 : 46;
        return [
          tubeSolid([{ y: 0, ro: 62, ri: lip }, { y: 5, ro: 62, ri: lip }, { y: 5, ro: 62, ri: gal }, { y: 9.6, ro: 62, ri: gal }], 360),
          radialSolid([{ y: 9.5, r: 62 }, { y: 13.2, r: 62 }, { y: 14, r: 61.2 }], 120)];
      }, pos: [0, 0, HUB_Z], explode: [0, -130, 0] });
    add({ id: 'boss', name: 'Bayonet boss (pedestal)', group: 'Mount', mat: 'brass', qty: 1, parent: 'pedestal', print: 'axis',
      note: 'Sits through the top plate onto the column. Three lugs, 4 mm thick. Three matching magnets in its crown.',
      shells: () => {
        const lug = th => LUG_ANGLES.some(a => Math.abs(angDiff(th, a)) < lugHalf) ? 52.5 : 45;
        const P = PED_TOP;
        return [radialSolid([{ y: PLATE_BOT, r: 62 }, { y: P, r: 62 }, { y: P, r: 45 }, { y: P + 5.3, r: 45 }, { y: P + 5.3, r: lug },
          { y: P + 9.0, r: lug }, { y: P + 9.0, r: 45 }, { y: P + 9.2, r: 44.2 }], 360)];
      }, explode: [0, 135, 0] });

    // ── Lid: the Sun ──
    const lidPoly = [[-(hwOut(LID_FRONT) + 4), LID_FRONT - LID_REAR], [hwOut(LID_FRONT) + 4, LID_FRONT - LID_REAR], [hwOut(LID_REAR) + 4, 0], [-(hwOut(LID_REAR) + 4), 0]];
    const lidProfile = [[0, 0], [0, 3], [-0.6, 4.5], [-2, 6], [-3.6, 7.2], [-5, 8.6], [-5.6, 10.2], [-6.4, 11.8], [-8, 13.2], [-10, 14]];
    add({ id: 'lid', name: 'Skylight cover', group: 'Lid', mat: 'walnut', qty: 1, parent: 'lid',
      note: 'Solid walnut, moulded edge. Four 6 × 2 mm magnets hold it to the roof; stands in the crest groove when raised.',
      shells: () => [profileSolid(lidPoly, lidProfile)], explode: [0, 0, 0] });
    const SUN_Z = (LID_FRONT - LID_REAR) / 2;
    add({ id: 'sun', name: 'The Sun — relief panel', group: 'Lid', mat: 'gilt', qty: 1, parent: 'lid', print: true,
      note: '150 × 100 mm bas-relief, 6 mm of relief on a 2.5 mm back. Print in resin at 0.05 mm, then gesso and water-gild.',
      shells: () => {
        const s = heightRect(150, 100, 0.5 / q, 2.5, sunHeight);
        return [s];
      }, pos: [0, LID_H - 0.4, SUN_Z], explode: [0, 45, 0] });
    add({ id: 'rest', name: 'Cover rest (cresting)', group: 'Ornament', mat: 'gilt', qty: 1, parent: 'viewer', print: true,
      note: 'Screwed to the roof behind the card slot. The raised cover stands in its 15 mm groove, leaning on the cresting.',
      shells: () => {
        const half = hwOut(RAIL_Z1) + 2;
        return [
          extrudeZY([[RAIL_Z0, ROOF_TOP], [RAIL_Z1, ROOF_TOP], [RAIL_Z1, ROOF_TOP + 4], [GROOVE_Z0, ROOF_TOP + 4], [GROOVE_Z0, ROOF_TOP + 10], [RAIL_Z0 + 1, ROOF_TOP + 10], [RAIL_Z0, ROOF_TOP + 9]], -half, half),
          extrudeXY(crestingOutline(half, ROOF_TOP + 3.9), [], GROOVE_Z1, RAIL_Z1)];
      }, explode: [0, 175, 30] });

    // ── Ornament on the body ──
    const endFrame = [[-1.5, 0], [9.5, 0], [9.5, 1.5], [8.8, 2.8], [7.6, 4.2], [5.8, 5.4], [3.8, 5.6], [2.2, 4.8], [1.2, 3.6], [0.4, 2.4], [-1.5, 2.4]];
    add({ id: 'end-frame', name: 'Ground-glass frame', group: 'Ornament', mat: 'gilt', qty: 1, parent: 'viewer', print: true,
      note: 'Bolection frame around the end window, where the double image appears.',
      shells: () => [sweepRing([[-78, -30.5], [78, -30.5], [78, 30.5], [-78, 30.5]], endFrame)],
      rot: [PI / 2, 0, 0], pos: [0, 60.5, L], explode: [0, 0, 140] });
    const wallAng = Math.atan2(HW1 - HW0, L);
    const medalZ = 92, medalY = 58;
    ['r', 'l'].forEach(side => {
      const sg = side === 'r' ? 1 : -1;
      const xs = hwOut(medalZ);
      add({ id: 'medal-' + side, name: 'Side medallion, ' + (side === 'r' ? 'right' : 'left'), group: 'Ornament', mat: 'gilt', qty: 1, parent: 'viewer', print: true,
        shared: 'medal', note: 'Laurel wreath around a small sun, 80 × 60 mm. Same file both sides.',
        shells: () => [heightOval(30, 40, Math.round(60 * q), Math.round(220 * q), 1.8, (x, z, r, t) => wreathHeight(z, -x, r, t))],
        rot: [0, sg > 0 ? wallAng : PI - wallAng, -PI / 2], rotOrder: 'YXZ', pos: [sg * xs, medalY, medalZ], explode: [sg * 150, 0, 0] });
    });

    // ── Card & ribbon (display) ──
    add({ id: 'card', name: 'Stereograph', group: 'Cards', mat: 'card', qty: 1, parent: 'card', buy: true, noExport: true,
      shells: () => [box(-CARD_W / 2, CARD_W / 2, -CARD_H / 2, CARD_H / 2, -0.5, 0.5)] });
    add({ id: 'ribbon', name: 'Silk lift ribbon', group: 'Viewer body', mat: 'ribbon', qty: 1, parent: 'viewer', buy: true, noExport: true,
      note: '10 mm silk ribbon, both ends pinned under the roof: it cradles the card; pull a loop to lift the card out.',
      shells: () => [box(-66, -56, CARD_BOTTOM - 0.4, 124, 153.4, 153.8), box(56, 66, CARD_BOTTOM - 0.4, 124, 153.4, 153.8),
        box(-66, 66, CARD_BOTTOM - 0.4, CARD_BOTTOM, 153.4, 156.6), box(-66, -56, 122.8, 123.2, 140, 153.8), box(56, 66, 122.8, 123.2, 140, 153.8)],
      explode: [0, 210, 0] });

    // ── Pedestal ──
    const baseRect = [[-CW, CZ0], [CW, CZ0], [CW, DF], [-CW, DF]];
    add({ id: 'plinth', name: 'Plinth', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      note: 'Walnut, 16 mm, ogee edge. Forms the carcase bottom.',
      shells: () => [profileSolid(baseRect, [[12, 0], [12, 3], [10.5, 4.8], [8, 6.3], [6, 8.2], [5, 10.5], [3.8, 12.8], [2, 14.8], [0, 16]])],
      explode: [0, -45, 0] });
    add({ id: 'base-side-r', name: 'Carcase side, right', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      shells: () => [box(CW - T, CW, 16, 121, CZ0, CZ1)], explode: [80, 0, 0] });
    add({ id: 'base-side-l', name: 'Carcase side, left', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      shells: () => [box(-CW, -CW + T, 16, 121, CZ0, CZ1)], explode: [-80, 0, 0] });
    add({ id: 'base-back', name: 'Carcase back', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      shells: () => [box(-CW + T, CW - T, 16, 121, CZ0, CZ0 + T)], explode: [0, 0, -80] });
    add({ id: 'base-top', name: 'Carcase top', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      note: 'Walnut, 15 mm, thumbnail edge. The column bolts through it.',
      shells: () => [profileSolid(baseRect, [[0, 121], [3, 122], [6, 123], [9, 124.5], [11.5, 127], [12.3, 130], [11.6, 133], [9.5, 135], [6, 136]])],
      explode: [0, 25, 0] });
    // drawer
    const DX = CW - T - 1, DI = DX - T;
    add({ id: 'drawer-front', name: 'Drawer front', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      note: 'Walnut, 19 mm, raised and bevelled field.',
      shells: () => [profileSolid([[-CW, -51.5], [CW, -51.5], [CW, 51.5], [-CW, 51.5]], [[0, 0], [0, 11], [-1.5, 13], [-6, 15.5], [-11, 17.5], [-13, 19]])],
      rot: [PI / 2, 0, 0], pos: [0, 68.5, CZ1], explode: [0, 0, 60] });
    add({ id: 'drawer-side-r', name: 'Drawer side, right', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      shells: () => [box(DI, DX, 17, 119, -107, CZ1)], explode: [55, 0, 0] });
    add({ id: 'drawer-side-l', name: 'Drawer side, left', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      shells: () => [box(-DX, -DI, 17, 119, -107, CZ1)], explode: [-55, 0, 0] });
    add({ id: 'drawer-back', name: 'Drawer back', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      shells: () => [box(-DI, DI, 17, 119, -107, -107 + T)], explode: [0, 0, -50] });
    add({ id: 'drawer-bottom', name: 'Drawer bottom', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      note: '6 mm walnut ply in grooves. Interior 183 × 198 × 94 mm: about 250 cards on their long edges.',
      shells: () => [box(-DI, DI, 19, 25, -107 + T, CZ1)], explode: [0, -30, 0] });
    add({ id: 'follower', name: 'Card follower', group: 'Drawer', mat: 'walnut', qty: 1, parent: 'drawer',
      note: 'Slides behind the cards to keep them upright. Finger notch on top.',
      shells: () => {
        const pts = [[-90.5, 25], [90.5, 25], [90.5, 95], [16, 95]];
        for (let i = 0; i <= 16; i++) { const a = i / 16 * PI; pts.push([16 * Math.cos(a), 95 - 12 * Math.sin(a)]); }
        pts.push([-90.5, 95]);
        return [extrudeXY(pts, [], -86, -70)];
      }, explode: [0, 70, 0] });
    add({ id: 'divider', name: 'Index divider', group: 'Drawer', mat: 'bone', qty: 2, parent: 'drawer', print: true, multi: [30, -18],
      note: 'Print as many as you like: 1.6 mm with a tab for a label.',
      shells: () => extrudeXYArr([[-89, 26], [89, 26], [89, 106], [-40, 106], [-44, 117], [-74, 117], [-78, 106], [-89, 106]], 0, 1.6),
      explode: [0, 90, 0] });
    add({ id: 'card-stack', name: 'Your stereographs', group: 'Cards', mat: 'stack', qty: 1, parent: 'drawer', buy: true, noExport: true,
      shells: () => [box(-89, 89, 25, 114, -69, 96)], explode: [0, 40, 0] });
    add({ id: 'knob', name: 'Drawer knob', group: 'Drawer', mat: 'gilt', qty: 1, parent: 'drawer', print: 'axis',
      note: 'Sunflower rosette and gadrooned knob; an M3 screw from inside the drawer.',
      shells: () => {
        const rings = knobRings();
        return [radialSolid(rings, 144)];
      }, rot: [PI / 2, 0, 0], pos: [0, 68.5, DF], explode: [0, 0, 110] });
    // base medallions
    [['r', 1], ['l', -1]].forEach(([s, sg]) => {
      add({ id: 'base-medal-' + s, name: 'Base medallion, ' + (s === 'r' ? 'right' : 'left'), group: 'Ornament', mat: 'gilt', qty: 1, parent: 'pedestal', print: true,
        shared: 'medal', note: 'Same wreath as the viewer sides.',
        shells: () => [heightOval(30, 40, Math.round(60 * q), Math.round(220 * q), 1.8, (x, z, r, t) => wreathHeight(z, -x, r, t))],
        rot: [0, sg > 0 ? 0 : PI, -PI / 2], rotOrder: 'YXZ', pos: [sg * CW, 68, -9], explode: [sg * 130, 0, 0] });
    });
    // column: base torus, gadrooned urn, Solomonic (twisted) shaft, acanthus bell, flared capital
    add({ id: 'column', name: 'Gilt column', group: 'Pedestal', mat: 'gilt', qty: 1, parent: 'pedestal', print: 'axis',
      note: 'Stand-in for your sculpted column, 205 mm tall: gadrooned urn, twisted Solomonic shaft, acanthus bell and capital. Drill through for an M8 rod.',
      shells: () => [radialSolid(columnRings(), Math.round(192 * q))], explode: [0, 60, 0] });
    for (let k = 0; k < 8; k++) {
      const upper = k >= 4, a = PI / 4 + (k % 4) * PI / 2;
      add({ id: 'console-' + k, name: 'Scroll console', group: 'Pedestal', mat: 'gilt', qty: 1, parent: 'pedestal', print: true, shared: 'console',
        note: 'Eight identical S-scrolls: four buttress the foot of the column, four (inverted) carry the top.',
        shells: () => [extrudeXY(consoleShape(), [], -7, 7)],
        rot: upper ? [PI, -a, 0] : [0, -a, 0], rotOrder: 'YXZ',
        pos: upper ? [0, PLATE_BOT + 138.5, 0] : [0, BASE_TOP - 139.5, 0],
        explode: [70 * Math.cos(a), upper ? 90 : 40, 70 * Math.sin(a)] });
    }
    const plate = PLATE_POLY;
    add({ id: 'top-plate', name: 'Pedestal top', group: 'Pedestal', mat: 'walnut', qty: 1, parent: 'pedestal',
      note: 'Walnut, 24 mm, canted corners. The brass boss drops through its 124 mm hole.',
      shells: () => [sweepRing(plate, [[-7, 0], [-6, 0], [-3, 1.5], [-1, 4], [0, 8], [0, 16], [-1, 20], [-3, 22.5], [-6, 24], [-7, 24]].map(([o, y]) => [o, PLATE_BOT + y])),
        extrudePlan(insetPoly(plate, 6.5), [circle(0, 0, 62, 96)], PLATE_BOT, 24)],
      explode: [0, 95, 0] });
    // expand arrays
    return parts;
  }
  function extrudeXYArr(pts, z0, z1) { return [extrudeXY(pts, [], z0, z1)]; }

  root.Helio = {
    buildParts, finalize,
    lib: { Shell, box, radialSolid, tubeSolid, extrudePlan, extrudeXY, extrudeZY, sweepRing, profileSolid, circle, insetPoly, transformShell, shellFromGeo, columnRings, knobRings, polyFrame },
    D: { PLATE_BOT, CW, CZ0, CZ1, DF, PLATE_POLY, NOUT: nOut, WALL_ANG: Math.atan2(HW1 - HW0, L), FLOOR_TOP, T, L, HW0, HW1, AXIS_Y, CARD_Z, LENS_Z, LENS_X, HUB_Z, PED_TOP, ROOF_TOP, WALL_TOP, CARD_BOTTOM, CARD_W, CARD_H,
      BASE_TOP, LID_FRONT, LID_REAR, LID_H, GROOVE_Z0, GROOVE_Z1, LOCK_TURN, LEAD: THREAD.pitch * THREAD.starts, hwOut, hwIn }
  };
})(typeof window !== 'undefined' ? window : globalThis);
