/* Heliotrope stereoscope — viewer, animation, STL export. */
(function () {
  const D = Helio.D;
  const PI = Math.PI, TAU = PI * 2;
  const $ = s => document.querySelector(s);
  const stage = $('#stage'), canvas = $('#c');

  // ───────────────────────────── renderer & scene ──────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.localClippingEnabled = true;
  const cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new THREE.RoomEnvironment(), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(30, 1, 5, 9000);
  camera.position.set(640, 560, 820);
  const controls = new THREE.OrbitControls(camera, canvas);
  controls.target.set(0, 250, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 120; controls.maxDistance = 2600;
  controls.maxPolarAngle = PI * 0.52;

  scene.add(new THREE.HemisphereLight(0xfff1dc, 0x2a1d14, 0.35));
  const key = new THREE.DirectionalLight(0xffe6c2, 1.55);
  key.position.set(420, 980, 520);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -420, right: 420, top: 420, bottom: -420, near: 200, far: 2400 });
  key.shadow.bias = -0.0004; key.shadow.normalBias = 0.6;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xbcd0ff, 0.45);
  rim.position.set(-600, 500, -700); scene.add(rim);
  const ground = new THREE.Mesh(new THREE.CircleGeometry(1400, 64), new THREE.ShadowMaterial({ opacity: 0.38 }));
  ground.rotation.x = -PI / 2; ground.receiveShadow = true; scene.add(ground);

  // ───────────────────────────── textures ───────────────────────────────────
  const tex = (cv, rep) => {
    const t = new THREE.CanvasTexture(cv); t.encoding = THREE.sRGBEncoding;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    if (rep) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
    return t;
  };
  function walnutCanvas() {
    const N = 1024, c = document.createElement('canvas'); c.width = c.height = N;
    const x = c.getContext('2d'), img = x.createImageData(N, N), d = img.data;
    const dark = [44, 27, 17], mid = [78, 50, 31], light = [106, 70, 43];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const w = i + 7 * Math.sin(j * 0.0045) + 3 * Math.sin(j * 0.017 + i * 0.006);
      const g = 0.5 + 0.5 * Math.sin(w * 0.21 + 1.6 * Math.sin(w * 0.027));
      const fig = 0.5 + 0.25 * Math.sin(i * 0.011 + 2 * Math.sin(j * 0.003)) + 0.25 * Math.sin(i * 0.037 + j * 0.002);
      const n = Math.abs((Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1);
      const t = Math.min(1, 0.3 * Math.pow(g, 3) + 0.55 * fig + 0.15 * n);
      const a = t < 0.5 ? dark : mid, b = t < 0.5 ? mid : light, tt = t < 0.5 ? t * 2 : (t - 0.5) * 2;
      const k = (j * N + i) * 4;
      d[k] = a[0] + (b[0] - a[0]) * tt; d[k + 1] = a[1] + (b[1] - a[1]) * tt; d[k + 2] = a[2] + (b[2] - a[2]) * tt; d[k + 3] = 255;
    }
    x.putImageData(img, 0, 0);
    return c;
  }
  // A Venetian view, drawn twice with parallax for the two eyes.
  function drawView(ctx, x0, y0, w, h, eye) {
    ctx.save();
    ctx.beginPath();
    const r = w / 2;
    ctx.moveTo(x0, y0 + h); ctx.lineTo(x0, y0 + r * 0.55);
    ctx.quadraticCurveTo(x0, y0, x0 + r, y0); ctx.quadraticCurveTo(x0 + w, y0, x0 + w, y0 + r * 0.55);
    ctx.lineTo(x0 + w, y0 + h); ctx.closePath(); ctx.clip();
    const sky = ctx.createLinearGradient(0, y0, 0, y0 + h * 0.6);
    sky.addColorStop(0, '#cdb48a'); sky.addColorStop(1, '#e6d3ad');
    ctx.fillStyle = sky; ctx.fillRect(x0, y0, w, h);
    const hz = y0 + h * 0.58;
    const px = (depth) => eye * depth;   // near things shift more
    // far: church of the Salute
    ctx.fillStyle = '#9d8563';
    const cx = x0 + w * 0.62 + px(2);
    ctx.fillRect(cx - 70, hz - 90, 140, 90);
    ctx.beginPath(); ctx.ellipse(cx, hz - 90, 58, 62, 0, PI, 0); ctx.fill();
    ctx.fillRect(cx - 6, hz - 175, 12, 28);
    ctx.beginPath(); ctx.ellipse(cx, hz - 150, 12, 14, 0, PI, 0); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx + 92, hz - 58, 26, 28, 0, PI, 0); ctx.fill();
    ctx.fillRect(cx + 66, hz - 58, 52, 58);
    for (let k = 0; k < 5; k++) { ctx.beginPath(); ctx.arc(cx - 64 + k * 32, hz - 92, 7, 0, TAU); ctx.fill(); }
    // palazzi on the left, mid distance
    const pl = [[0, 150, 70], [150, 90, 110], [240, 70, 88]];
    pl.forEach(([ox, pw, ph], k) => {
      const bx = x0 - 20 + ox + px(8);
      ctx.fillStyle = ['#7c6647', '#8a7352', '#6f5a3e'][k];
      ctx.fillRect(bx, hz - ph, pw, ph + 4);
      ctx.fillStyle = '#4c3c28';
      for (let row = 0; row < 3; row++) for (let col = 0; col < Math.floor(pw / 22); col++) {
        const wx = bx + 8 + col * 22, wy = hz - ph + 14 + row * 24;
        ctx.fillRect(wx, wy + 5, 9, 13); ctx.beginPath(); ctx.arc(wx + 4.5, wy + 5, 4.5, PI, 0); ctx.fill();
      }
    });
    // water
    const water = ctx.createLinearGradient(0, hz, 0, y0 + h);
    water.addColorStop(0, '#b29873'); water.addColorStop(1, '#6f5a40');
    ctx.fillStyle = water; ctx.fillRect(x0, hz, w, h);
    ctx.strokeStyle = 'rgba(230,210,170,.35)'; ctx.lineWidth = 1.2;
    for (let k = 0; k < 40; k++) {
      const yy = hz + 6 + k * k * 0.16, xx = x0 + ((k * 97) % w);
      ctx.beginPath(); ctx.moveTo(xx + px(k * 0.4), yy); ctx.lineTo(xx + 30 + k * 2 + px(k * 0.4), yy); ctx.stroke();
    }
    // gondola, near
    const gx = x0 + w * 0.36 + px(22), gy = y0 + h * 0.8;
    ctx.fillStyle = '#231a12';
    ctx.beginPath(); ctx.moveTo(gx - 130, gy - 26); ctx.quadraticCurveTo(gx - 40, gy + 12, gx + 120, gy - 4);
    ctx.lineTo(gx + 146, gy - 38); ctx.lineTo(gx + 138, gy - 6); ctx.quadraticCurveTo(gx - 20, gy + 26, gx - 130, gy - 26); ctx.fill();
    ctx.fillRect(gx + 70, gy - 70, 5, 64);
    ctx.beginPath(); ctx.arc(gx + 72, gy - 76, 7, 0, TAU); ctx.fill();
    // mooring poles, nearest
    [[0.08, 34], [0.14, 30], [0.88, 36]].forEach(([fx, dep]) => {
      const bx = x0 + w * fx + px(dep);
      ctx.fillStyle = '#2a1f15'; ctx.fillRect(bx, y0 + h * 0.28, 11, h);
      ctx.fillStyle = '#d8c39c';
      for (let k = 0; k < 6; k++) ctx.fillRect(bx, y0 + h * 0.3 + k * 38, 11, 14);
    });
    // tone, vignette
    const vg = ctx.createRadialGradient(x0 + w / 2, y0 + h / 2, h * 0.25, x0 + w / 2, y0 + h / 2, h * 0.85);
    vg.addColorStop(0, 'rgba(60,40,20,0)'); vg.addColorStop(1, 'rgba(60,40,20,.55)');
    ctx.fillStyle = vg; ctx.fillRect(x0, y0, w, h);
    ctx.restore();
  }
  function cardCanvas() {
    const W = 1780, H = 890, c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d');
    x.fillStyle = '#e4d3ad'; x.fillRect(0, 0, W, H);
    x.fillStyle = '#d6c193'; x.fillRect(18, 18, W - 36, H - 36);
    x.fillStyle = '#e8d9b5'; x.fillRect(26, 26, W - 52, H - 52);
    drawView(x, 110, 60, 760, 700, -1);
    drawView(x, 910, 60, 760, 700, 1);
    x.fillStyle = '#4a3a26'; x.textAlign = 'center';
    x.font = 'italic 34px "IM Fell English", Georgia, serif';
    x.fillText('The Grand Canal and the Salute, Venice.', W / 2, 818);
    x.font = '24px "IM Fell English SC", Georgia, serif';
    x.fillText('No. 214', 180, 818); x.fillText('Stereoscopic View', W - 220, 818);
    for (let k = 0; k < 9000; k++) { x.fillStyle = `rgba(80,55,30,${Math.random() * 0.06})`; x.fillRect(Math.random() * W, Math.random() * H, 2, 2); }
    return c;
  }
  function cardBackCanvas() {
    const c = document.createElement('canvas'); c.width = 890; c.height = 445;
    const x = c.getContext('2d');
    x.fillStyle = '#d9c7a0'; x.fillRect(0, 0, 890, 445);
    x.fillStyle = 'rgba(80,60,40,.45)'; x.textAlign = 'center';
    x.font = '26px "IM Fell English SC", Georgia, serif'; x.fillText('Venice', 445, 150);
    x.font = 'italic 18px "IM Fell English", Georgia, serif';
    ['Here the Grand Canal opens toward the lagoon; the great', 'dome of Santa Maria della Salute rises on the right.']
      .forEach((l, i) => x.fillText(l, 445, 200 + i * 28));
    return c;
  }
  function stackCanvas() {
    const c = document.createElement('canvas'); c.width = 512; c.height = 64;
    const x = c.getContext('2d');
    for (let i = 0; i < 512; i += 2) {
      const v = 190 + Math.random() * 45;
      x.fillStyle = `rgb(${v},${v * 0.9},${v * 0.72})`; x.fillRect(i, 0, 2, 64);
      if (Math.random() < 0.08) { x.fillStyle = 'rgba(60,40,25,.5)'; x.fillRect(i, 0, 1, 64); }
    }
    return c;
  }
  // Two upside-down images of the room, as the lenses throw them onto the ground glass.
  function glassCanvas(loaded) {
    const W = 800, H = 320, c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d');
    x.fillStyle = loaded ? '#2c231b' : '#1c1611'; x.fillRect(0, 0, W, H);
    if (!loaded) {
      x.filter = 'blur(7px)';
      [-1, 1].forEach(sd => {
        const cx = W / 2 + sd * 190;
        x.save(); x.translate(cx, H / 2); x.scale(-1, -1);
        const g = x.createRadialGradient(0, 0, 10, 0, 0, 180);
        g.addColorStop(0, '#6b5238'); g.addColorStop(1, '#1c1611');
        x.fillStyle = g; x.fillRect(-200, -160, 400, 320);
        x.fillStyle = '#f4e3bf'; x.fillRect(-120, -110, 90, 130);
        x.fillStyle = '#6b5238'; x.fillRect(-78, -110, 6, 130); x.fillRect(-120, -48, 90, 6);
        x.fillStyle = '#ffcf7a'; x.beginPath(); x.arc(70, -20, 16, 0, TAU); x.fill();
        x.fillStyle = '#3a2a1c'; x.fillRect(-190, 70, 380, 90);
        x.fillStyle = '#4a3522'; x.fillRect(40, 0, 60, 70);
        x.restore();
      });
      x.filter = 'none';
    } else {
      x.fillStyle = 'rgba(210,180,130,.08)'; x.fillRect(20, 30, W - 40, H - 60);
    }
    for (let k = 0; k < 14000; k++) { x.fillStyle = `rgba(255,240,215,${Math.random() * 0.05})`; x.fillRect(Math.random() * W, Math.random() * H, 1.5, 1.5); }
    return c;
  }

  const walnutTex = tex(walnutCanvas(), true);
  const cardCv = cardCanvas();
  const cardTex = tex(cardCv), cardBackTex = tex(cardBackCanvas()), stackTex = tex(stackCanvas());
  const glassTexEmpty = tex(glassCanvas(false)), glassTexLoaded = tex(glassCanvas(true));

  // ───────────────────────────── materials ─────────────────────────────────
  const MAT = {
    walnut: () => new THREE.MeshPhysicalMaterial({ map: walnutTex, roughness: 0.5, clearcoat: 0.35, clearcoatRoughness: 0.35, envMapIntensity: 0.55 }),
    gilt: () => new THREE.MeshStandardMaterial({ color: 0xd49c42, metalness: 1, roughness: 0.32, envMapIntensity: 1.0 }),
    brass: () => new THREE.MeshStandardMaterial({ color: 0xc19650, metalness: 1, roughness: 0.4 }),
    black: () => new THREE.MeshStandardMaterial({ color: 0x15120f, roughness: 0.85 }),
    groundglass: () => new THREE.MeshPhysicalMaterial({ color: 0xf2ede2, roughness: 0.9, transparent: true, opacity: 0.5, depthWrite: false }),
    lens: () => new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.03, transmission: 1, thickness: 3, ior: 1.5, transparent: true }),
    bone: () => new THREE.MeshStandardMaterial({ color: 0xe9dcc0, roughness: 0.6 }),
    ribbon: () => new THREE.MeshStandardMaterial({ color: 0x5e0f1c, roughness: 0.5 }),
    card: () => { const edge = new THREE.MeshStandardMaterial({ color: 0xcbb68b, roughness: 0.8 });
      return [edge, edge, edge, edge, new THREE.MeshStandardMaterial({ map: cardTex, roughness: 0.55 }), new THREE.MeshStandardMaterial({ map: cardBackTex, roughness: 0.8 })]; },
    stack: () => { const s = new THREE.MeshStandardMaterial({ map: stackTex, roughness: 0.8 });
      return [s, s, s, s, new THREE.MeshStandardMaterial({ map: cardTex, roughness: 0.6 }), s]; },
  };
  const SWATCH = { walnut: '#6a4428', gilt: '#d9ad55', brass: '#b58c4a', black: '#1d1915', groundglass: '#d8d2c4', lens: '#9fc3cf', bone: '#e9dcc0', ribbon: '#7c1827', card: '#e4d3ad', stack: '#e4d3ad' };
  const MATNAME = { walnut: 'walnut', gilt: 'gilt', brass: 'brass', black: 'matte black', groundglass: 'ground glass', lens: 'glass', bone: 'bone', ribbon: 'silk', card: 'card', stack: 'cards' };

  // ───────────────────────────── scene graph ───────────────────────────────
  const pedestal = new THREE.Group(); scene.add(pedestal);
  const drawer = new THREE.Group(); pedestal.add(drawer);
  const pivot = new THREE.Group(); pivot.position.y = D.PED_TOP; scene.add(pivot);
  const viewer = new THREE.Group(); viewer.position.z = -D.HUB_Z; pivot.add(viewer);
  const lid = new THREE.Group(); viewer.add(lid);
  const card = new THREE.Group(); viewer.add(card);
  const PARENT = { pedestal, drawer, viewer, lid, card };

  const parts = Helio.buildParts();
  const meshes = [];            // {mesh, part, base: Vector3, explode: Vector3, kind}
  const rows = new Map();       // key -> {parts:[], meshes:[], el}
  let glassPlane = null;

  function placeMesh(part, geo, mat, extra = {}) {
    const m = new THREE.Mesh(geo, mat);
    m.castShadow = !['groundglass', 'lens'].includes(part.mat);
    m.receiveShadow = true;
    const pos = extra.pos || part.pos || [0, 0, 0];
    m.position.set(pos[0], pos[1], pos[2]);
    if (part.rot) m.rotation.set(part.rot[0], part.rot[1], part.rot[2], part.rotOrder || 'XYZ');
    if (extra.rot) m.rotation.set(extra.rot[0], extra.rot[1], extra.rot[2], 'XYZ');
    PARENT[part.parent].add(m);
    const rec = { mesh: m, part, base: m.position.clone(), explode: new THREE.Vector3(...(part.explode || [0, 0, 0])), kind: extra.kind };
    meshes.push(rec);
    m.userData.key = part.shared || part.id;
    const r = rows.get(m.userData.key); r.meshes.push(m);
    return rec;
  }

  async function build() {
    const bar = $('#lbar');
    parts.forEach(p => {
      const k = p.shared || p.id;
      if (!rows.has(k)) rows.set(k, { key: k, parts: [], meshes: [] });
      rows.get(k).parts.push(p);
    });
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      bar.style.width = (100 * (i + 1) / parts.length) + '%';
      await new Promise(r => setTimeout(r, 0));
      if (p.id === 'card') {
        placeMesh(p, new THREE.BoxGeometry(D.CARD_W, D.CARD_H, 1), MAT.card(), { rot: [0, PI, 0] });
        continue;
      }
      if (p.id === 'card-stack') {
        const g = new THREE.BoxGeometry(178, 89, 165); g.translate(0, 69.5, 13.5);
        placeMesh(p, g, MAT.stack());
        continue;
      }
      const shells = p.shells();
      p._shells = shells;
      const geo = Helio.finalize(shells);
      if (p.pair) {
        [-1, 1].forEach(sg => placeMesh(p, geo, MAT[p.mat](), { pos: [sg * D.LENS_X, D.AXIS_Y, 0], kind: p.id }));
      } else if (p.multi) {
        p.multi.forEach(z => placeMesh(p, geo, MAT[p.mat](), { pos: [0, 0, z] }));
      } else {
        const rec = placeMesh(p, geo, MAT[p.mat]());
        if (p.id === 'ground-glass') {
          glassPlane = new THREE.Mesh(new THREE.PlaneGeometry(160, 64), new THREE.MeshBasicMaterial({ map: glassTexEmpty, toneMapped: false }));
          glassPlane.position.set(0, 60.5, 161.1);
          rec.mesh.add(glassPlane);
        }
      }
    }
    buildPartsList();
    $('#loading').hidden = true;
  }

  // ───────────────────────────── state & poses ─────────────────────────────
  const REST_PSI = PI;
  const S = { drawer: 0, card: 3, lid: 1, focus: 0, psi: REST_PSI, lift: 0, explode: 0 };
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const lidDockY = D.ROOF_TOP + 4 + (D.LID_REAR - D.LID_FRONT), lidDockZ = D.GROOVE_Z1 - 0.5;
  const LIDK = [
    { p: [0, D.ROOF_TOP, D.LID_REAR], r: 0 },
    { p: [0, D.ROOF_TOP + 45, D.LID_REAR], r: 0 },
    { p: [0, lidDockY + 45, lidDockZ], r: -PI / 2 },
    { p: [0, lidDockY, lidDockZ], r: -PI / 2 }];
  function keyframes(K, u) {
    const n = K.length - 1, s = Math.min(n - 1e-6, Math.max(0, u * n)), i = Math.floor(s), t = ease(s - i);
    return { p: K[i].p.map((v, k) => lerp(v, K[i + 1].p[k], t)), r: lerp(K[i].r, K[i + 1].r, t) };
  }
  const tmpV = new THREE.Vector3();
  function cardPath(c) {
    // world points in the drawer → viewer-local
    scene.updateMatrixWorld(true);
    const inDrawer = drawer.localToWorld(tmpV.set(0, 69.5, 98.5)).clone();
    const lifted = inDrawer.clone().add(new THREE.Vector3(0, 230, 0));
    const P0 = viewer.worldToLocal(inDrawer.clone()), P1 = viewer.worldToLocal(lifted.clone());
    const K = [P0.toArray(), P1.toArray(), [0, D.AXIS_Y + 175, D.CARD_Z], [0, D.AXIS_Y, D.CARD_Z]];
    const n = 3, s = Math.min(n - 1e-6, Math.max(0, c)), i = Math.floor(s), t = s - i;
    const tt = i === 2 ? t : ease(t);
    return K[i].map((v, k) => lerp(v, K[i + 1][k], tt));
  }
  function apply() {
    const e = S.explode;
    for (const r of meshes) {
      r.mesh.position.copy(r.base).addScaledVector(r.explode, e);
      if (r.kind === 'draw-tube' || r.kind === 'lens' || r.kind === 'lens-ring') {
        r.mesh.position.z += D.LENS_Z + S.focus;
        r.mesh.rotation.set(-PI / 2, -TAU * S.focus / D.LEAD, 0);
      } else if (r.kind === 'sleeve') {
        r.mesh.rotation.set(-PI / 2, 0, 0);
      }
    }
    pivot.rotation.y = S.psi;
    pivot.position.y = D.PED_TOP + S.lift + 260 * e;
    drawer.position.z = S.drawer * 170 + 150 * e;
    const lk = keyframes(LIDK, e > 0 ? 0 : S.lid);
    lid.position.set(lk.p[0], lk.p[1] + 230 * e, lk.p[2]);
    lid.rotation.x = lk.r;
    viewer.updateMatrixWorld(true);
    const cp = cardPath(e > 0 ? 3 : S.card);
    card.position.set(cp[0], cp[1] + 290 * e, cp[2]);
    card.visible = !(S.card < 0.02 && S.drawer < 0.02 && e === 0);
    if (glassPlane) glassPlane.material.map = (S.card > 2.9 && e === 0) ? glassTexLoaded : glassTexEmpty;
  }

  // tweening
  let tween = null;
  function animateTo(target, dur = 1600, camTo) {
    const from = { ...S };
    const camFrom = { p: camera.position.clone(), t: controls.target.clone() };
    tween = { from, target, t0: performance.now(), dur, camFrom, camTo };
  }
  function stepTween(now) {
    if (!tween) return;
    const u = Math.min(1, (now - tween.t0) / tween.dur);
    for (const k in tween.target) S[k] = lerp(tween.from[k], tween.target[k], k === 'card' || k === 'lid' ? u : ease(u));
    if (tween.camTo) {
      const cu = ease(Math.min(1, (now - tween.t0) / Math.min(tween.dur, 1500)));
      camera.position.lerpVectors(tween.camFrom.p, new THREE.Vector3(...tween.camTo.p), cu);
      controls.target.lerpVectors(tween.camFrom.t, new THREE.Vector3(...tween.camTo.t), cu);
    }
    if (u >= 1) { const done = tween.onDone; tween = null; if (done) done(); }
  }
  function flyTo(p, t, dur = 1300) { animateTo({}, dur, { p, t }); }

  // ghosting for the mount view
  function ghost(on, keep) {
    for (const r of meshes) {
      const k = r.mesh.userData.key;
      const mats = Array.isArray(r.mesh.material) ? r.mesh.material : [r.mesh.material];
      const g = on && !keep.includes(k);
      mats.forEach(m => {
        if (m.userData.baseOpacity === undefined) { m.userData.baseOpacity = m.opacity; m.userData.baseTransparent = m.transparent; m.userData.baseDW = m.depthWrite; }
        m.transparent = g ? true : m.userData.baseTransparent;
        m.opacity = g ? 0.07 : m.userData.baseOpacity;
        m.depthWrite = g ? false : m.userData.baseDW;
        m.needsUpdate = true;
      });
      r.mesh.castShadow = !g && !['groundglass', 'lens'].includes(r.part.mat);
    }
    if (glassPlane) glassPlane.visible = !on;
  }

  // ───────────────────────────── modes ─────────────────────────────────────
  const caption = $('#caption'), controlsEl = $('#controls'), eyeview = $('#eyeview');
  let mode = 'assembled', mountT0 = 0, ritualStep = 0;
  const CAM = {
    assembled: { p: [620, 560, 800], t: [0, 255, 0] },
    exploded: { p: [1450, 1200, 1750], t: [0, 440, 0] },
    mount: { p: [400, 560, 520], t: [0, 270, 0] },
  };
  function setCaption(k, h, p) {
    if (!h) { caption.hidden = true; return; }
    caption.hidden = false;
    caption.innerHTML = (k ? `<span class="k">${k}</span>` : '') + `<h3>${h}</h3><p>${p}</p>`;
  }
  const seg = (id, label, opts, val) => `<div class="ctl-row"><span class="lab">${label}</span><div class="seg" id="${id}">` +
    opts.map(([v, l]) => `<button data-v="${v}" aria-pressed="${v == val}">${l}</button>`).join('') + '</div></div>';
  function wireSeg(id, fn) {
    const el = document.getElementById(id);
    el.addEventListener('click', ev => {
      const b = ev.target.closest('button'); if (!b) return;
      el.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
      fn(parseFloat(b.dataset.v));
    });
  }
  const diopters = f => 1000 / 150 - 1000 / (150 - f);
  function setMode(m) {
    mode = m;
    document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === m));
    ghost(m === 'mount', ['socket', 'boss', 'top-plate']);
    caption.dataset.idx = '';
    meshes.filter(r => r.part.id === 'socket').forEach(r => {
      r.mesh.material.clippingPlanes = m === 'mount' ? [cutPlane] : [];
      r.mesh.material.side = m === 'mount' ? THREE.DoubleSide : THREE.FrontSide;
      r.mesh.material.needsUpdate = true;
    });
    eyeview.hidden = true;
    tween = null;
    if (m === 'assembled') {
      Object.assign(S, { explode: 0, psi: REST_PSI, lift: 0 });
      controlsEl.innerHTML = `<h2>Assembled</h2><div class="ctl">
        ${seg('s-lid', 'Skylight cover', [[0, 'Closed'], [1, 'Raised']], S.lid > 0.5 ? 1 : 0)}
        ${seg('s-card', 'Card', [[0, 'In the drawer'], [3, 'In the viewer']], S.card > 1.5 ? 3 : 0)}
        ${seg('s-drawer', 'Drawer', [[0, 'Closed'], [1, 'Open']], S.drawer > 0.5 ? 1 : 0)}
        <div class="ctl-row"><label for="s-focus">Focus <output id="o-focus"></output></label>
          <input type="range" id="s-focus" min="-8" max="15" step="0.5" value="${S.focus}"></div>
        <p class="hint">Drag to orbit, scroll or pinch to zoom, right-drag to pan. Turn the focus to watch both helicoids screw in.</p></div>`;
      wireSeg('s-lid', v => animateTo({ lid: v }, 2200));
      wireSeg('s-card', v => animateTo({ card: v, drawer: 1 }, 2400));
      wireSeg('s-drawer', v => animateTo({ drawer: v }, 1200));
      const fr = $('#s-focus'), fo = $('#o-focus');
      const showF = () => { const d = diopters(+fr.value); fo.textContent = `${(+fr.value > 0 ? '+' : '') + fr.value} mm · ${d > 0.005 ? '+' : d < -0.005 ? '−' : ''}${Math.abs(d).toFixed(2)} D`; };
      fr.addEventListener('input', () => { S.focus = +fr.value; showF(); }); showF();
      setCaption();
      flyTo(CAM.assembled.p, CAM.assembled.t);
    } else if (m === 'exploded') {
      controlsEl.innerHTML = `<h2>Exploded</h2><div class="ctl">
        <div class="ctl-row"><label for="s-exp">Separation <output id="o-exp"></output></label>
          <input type="range" id="s-exp" min="0" max="1" step="0.01" value="1"></div>
        <p class="hint">Every part moves out along the direction it is assembled. Click any piece to find it in the list.</p></div>`;
      const ex = $('#s-exp'), eo = $('#o-exp');
      ex.addEventListener('input', () => { tween = null; S.explode = +ex.value; eo.textContent = Math.round(ex.value * 100) + '%'; });
      eo.textContent = '100%';
      animateTo({ explode: 1, psi: REST_PSI, lift: 0 }, 1500, CAM.exploded);
      setCaption();
    } else if (m === 'mount') {
      Object.assign(S, { explode: 0 });
      controlsEl.innerHTML = `<h2>Bayonet mount</h2><div class="ctl">
        <p class="hint" style="color:var(--muted)">The viewer is ghosted and its brass socket cut in half, so you can see the three lugs on the pedestal boss pass through the notches and turn under the lip. The cycle repeats.</p>
        <div class="nav"><button class="btn ghost" id="m-pause">Pause</button></div></div>`;
      let paused = false;
      $('#m-pause').addEventListener('click', e => { paused = !paused; e.target.textContent = paused ? 'Play' : 'Pause'; mountPaused = paused; if (!paused) mountT0 = performance.now() - mountClock; });
      mountPaused = false; mountT0 = performance.now();
      flyTo(CAM.mount.p, CAM.mount.t);
    } else if (m === 'ritual') {
      S.explode = 0;
      controlsEl.innerHTML = `<h2>Using it</h2><ol class="steps" id="steps">${RITUAL.map((s, i) =>
        `<li><button data-i="${i}"><b>${i + 1}</b><span>${s.h}</span></button></li>`).join('')}</ol>
        <div class="nav"><button class="btn ghost" id="r-prev">Back</button><button class="btn" id="r-next">Next step</button></div>`;
      $('#steps').addEventListener('click', e => { const b = e.target.closest('button'); if (b) goStep(+b.dataset.i); });
      $('#r-prev').addEventListener('click', () => goStep(Math.max(0, ritualStep - 1)));
      $('#r-next').addEventListener('click', () => goStep((ritualStep + 1) % RITUAL.length));
      goStep(0, true);
    }
  }
  let mountPaused = false, mountClock = 0;
  const MOUNT = [
    [0.0, 'Locked', 'The three lugs sit under the socket lip. The viewer cannot lift, slide or tip off; magnets hold it square.'],
    [1.4, 'Turn a twelfth anticlockwise', 'Holding the body under the cornice, turn it 30°. The lugs slide along the gallery until they reach the notches.'],
    [3.0, 'Lift', 'The lugs pass up through the notches. The viewer comes away and stands flat on any table.'],
    [5.2, 'Set it down turned', 'Lower it with the eyepieces turned 30° to the left. It drops 9.5 mm onto the boss when the notches find the lugs.'],
    [6.6, 'Turn clockwise to lock', 'Turn until it stops square to the pedestal. The stop sits in the socket, so you cannot over-turn it.']];
  function mountPose(t) {
    const T = 8.4, s = t % T;
    const seg = (a, b) => ease(Math.min(1, Math.max(0, (s - a) / (b - a))));
    const turnOut = seg(1.4, 2.6), up = seg(3.0, 4.1), down = seg(5.2, 6.3), turnIn = seg(6.6, 7.7);
    S.psi = REST_PSI + D.LOCK_TURN * (turnOut - turnIn);
    S.lift = 70 * (up - down);
    let idx = 0; MOUNT.forEach((m, i) => { if (s >= m[0]) idx = i; });
    if (s > 7.7) idx = 0;
    if (mode === 'mount' && caption.dataset.idx !== String(idx)) {
      caption.dataset.idx = idx;
      setCaption('Bayonet · ' + (idx + 1) + ' of 5', MOUNT[idx][1], MOUNT[idx][2]);
    }
  }

  const RITUAL = [
    { h: 'At rest', p: 'The Sun covers the skylight. On the far end, the ground glass shows two small upside-down images of the room. With no card in place, the lenses work as a camera obscura.',
      s: { drawer: 0, card: 0, lid: 0, focus: 0 }, cam: { p: [-560, 560, -780], t: [0, 290, -40] } },
    { h: 'Choose a card', p: 'Draw out the drawer. About 250 cards stand on their long edges, picture toward you, held up by a sliding follower. Lift one out by its top edge.',
      s: { drawer: 1, card: 1, lid: 0 }, cam: { p: [520, 520, 900], t: [0, 230, 130] } },
    { h: 'Raise the Sun', p: 'Lift the cover with both hands, tip it toward you until it stands, and set its lower edge in the crest at the back. The Sun now faces you, and daylight falls through the skylight.',
      s: { lid: 1 }, cam: { p: [560, 640, 620], t: [0, 360, 0] } },
    { h: 'Set the card', p: 'Drop the card picture-first into the slot in front of the Sun. It rides down onto the silk ribbon and stops on the gate sill, centred on both lenses. Slide the drawer home.',
      s: { card: 3, drawer: 0 }, cam: { p: [470, 620, 330], t: [0, 340, -40] } },
    { h: 'Focus', p: 'Turn each knurled ring. The helicoid moves the lens up to 15 mm toward the card for short sight, or 8 mm away. Each eye focuses separately.',
      s: { focus: 9 }, cam: { p: [300, 380, 430], t: [0, 314, 100] } },
    { h: 'Look', p: 'Rest your brow on the eyecups. The two pictures merge into one view in depth, lit from above, framed by the arch of the mask.',
      s: {}, cam: { p: [0, 345, 380], t: [0, 314, 0] }, eye: true },
    { h: 'Put it away', p: 'Pull a ribbon loop and the card rises from the slot. Return it to the drawer and lay the Sun back over the skylight, where four magnets seat it.',
      s: { card: 0, drawer: 1, lid: 0, focus: 0 }, cam: { p: [560, 560, 860], t: [0, 230, 60] } }];
  function goStep(i, instant) {
    ritualStep = i;
    const st = RITUAL[i];
    document.querySelectorAll('#steps button').forEach((b, k) => { if (k === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
    setCaption('Step ' + (i + 1) + ' of ' + RITUAL.length, st.h, st.p);
    eyeview.hidden = !st.eye;
    if (st.eye) drawEyeView();
    const target = { psi: REST_PSI, lift: 0, explode: 0, ...st.s };
    if (i === 0) Object.assign(target, { card: 0, drawer: 0, lid: 0, focus: 0 });
    animateTo(target, instant ? 1400 : 2600, st.cam);
  }
  function drawEyeView() {
    const cv = eyeview.querySelector('canvas'), x = cv.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, cv.width, cv.height);
    x.save();
    x.beginPath(); x.ellipse(cv.width / 2, cv.height / 2, cv.width * 0.47, cv.height * 0.46, 0, 0, TAU); x.clip();
    x.drawImage(cardCv, 110, 40, 760, 720, cv.width / 2 - 230, cv.height / 2 - 218, 460, 436);
    const g = x.createRadialGradient(cv.width / 2, cv.height / 2, 120, cv.width / 2, cv.height / 2, cv.width * 0.5);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,.85)');
    x.fillStyle = g; x.fillRect(0, 0, cv.width, cv.height);
    x.restore();
  }

  // ───────────────────────────── picking & parts list ──────────────────────
  let selKey = null;
  function select(key, scroll) {
    selKey = key;
    for (const r of meshes) {
      const on = r.mesh.userData.key === key;
      const mats = Array.isArray(r.mesh.material) ? r.mesh.material : [r.mesh.material];
      mats.forEach(m => { if (m.emissive) m.emissive.setHex(on ? 0x5a3a08 : 0x000000); });
    }
    rows.forEach(r => r.el && r.el.classList.toggle('sel', r.key === key));
    const picked = $('#picked');
    if (key) {
      const r = rows.get(key), p = r.parts[0];
      picked.hidden = false;
      picked.innerHTML = `<b>${esc(rowName(r))}</b> · ${MATNAME[p.mat]}${rowQty(r) > 1 ? ' · ×' + rowQty(r) : ''}`;
      if (scroll && r.el) r.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } else picked.hidden = true;
  }
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const rowQty = r => r.parts.reduce((a, p) => a + (p.qty || 1) * (p.multi ? 1 : 1), 0);
  const rowName = r => r.parts.length > 1 ? r.parts[0].name.replace(/, (right|left)$/, '') : r.parts[0].name;
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let down = null;
  canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  canvas.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const rc = canvas.getBoundingClientRect();
    ndc.set((e.clientX - rc.left) / rc.width * 2 - 1, -(e.clientY - rc.top) / rc.height * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(meshes.map(r => r.mesh).filter(m => m.visible), false)
      .filter(h => !(h.object.material.opacity < 0.2));
    select(hits.length ? hits[0].object.userData.key : null, true);
  });

  function printFrame(shells, mode) {
    let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    shells.forEach(s => { for (let i = 0; i < s.p.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], s.p[i + k]); mx[k] = Math.max(mx[k], s.p[i + k]); } });
    const ext = mx.map((v, k) => v - mn[k]);
    let map;   // returns [X,Y,Z] with Z up
    let small = ext.indexOf(Math.min(...ext));
    if (mode === 'axis') small = 1;
    if (small === 1) map = (x, y, z) => [x, -z, y];
    else if (small === 2) map = (x, y, z) => [x, y, z];
    else map = (x, y, z) => [y, z, x];
    const a = map(mn[0], mn[1], mn[2]), b = map(mx[0], mx[1], mx[2]);
    const lo = a.map((v, k) => Math.min(v, b[k])), hi = a.map((v, k) => Math.max(v, b[k]));
    const off = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]];
    return { map, off, size: hi.map((v, k) => v - lo[k]) };
  }
  function stlBytes(p) {
    const shells = p._shells || p.shells();
    const pf = printFrame(shells, p.print);
    let n = 0; shells.forEach(s => n += s.p.length / 9);
    const buf = new ArrayBuffer(84 + n * 50), dv = new DataView(buf);
    const head = `Heliotrope stereoscope - ${p.name} - mm`;
    for (let i = 0; i < 80; i++) dv.setUint8(i, i < head.length ? head.charCodeAt(i) & 127 : 32);
    let o = 84, count = 0;
    shells.forEach(s => {
      const P = s.p;
      for (let i = 0; i < P.length; i += 9) {
        const v = [0, 1, 2].map(k => { const m = pf.map(P[i + k * 3], P[i + k * 3 + 1], P[i + k * 3 + 2]); return [m[0] + pf.off[0], m[1] + pf.off[1], m[2] + pf.off[2]]; });
        const ax = v[1][0] - v[0][0], ay = v[1][1] - v[0][1], az = v[1][2] - v[0][2];
        const bx = v[2][0] - v[0][0], by = v[2][1] - v[0][1], bz = v[2][2] - v[0][2];
        let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        const l = Math.hypot(nx, ny, nz);
        if (l < 1e-9) continue;
        nx /= l; ny /= l; nz /= l;
        dv.setFloat32(o, nx, true); dv.setFloat32(o + 4, ny, true); dv.setFloat32(o + 8, nz, true); o += 12;
        for (let k = 0; k < 3; k++) { dv.setFloat32(o, v[k][0], true); dv.setFloat32(o + 4, v[k][1], true); dv.setFloat32(o + 8, v[k][2], true); o += 12; }
        dv.setUint16(o, 0, true); o += 2; count++;
      }
    });
    dv.setUint32(80, count, true);
    return new Uint8Array(buf, 0, 84 + count * 50);
  }
  // minimal store-only ZIP
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  const crc32 = u8 => { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  function zip(files) {
    const enc = new TextEncoder(), chunks = [], central = [];
    let off = 0;
    files.forEach(f => {
      const name = enc.encode(f.name), data = f.data, crc = crc32(data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(8, 0, true);
      h.setUint16(10, 0, true); h.setUint16(12, 0x21, true);
      h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true);
      h.setUint16(26, name.length, true);
      chunks.push(new Uint8Array(h.buffer), name, data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true);
      c.setUint16(14, 0x21, true); c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true);
      c.setUint16(28, name.length, true); c.setUint32(42, off, true);
      central.push(new Uint8Array(c.buffer), name);
      off += 30 + name.length + data.length;
    });
    const csize = central.reduce((a, c) => a + c.length, 0);
    const e = new DataView(new ArrayBuffer(22));
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
    e.setUint32(12, csize, true); e.setUint32(16, off, true);
    return new Blob([...chunks, ...central, new Uint8Array(e.buffer)], { type: 'application/zip' });
  }

  // downloads: the artifact viewer only saves via the downloads capability
  let dlCap;
  const dlReady = (window.claude && window.claude.use) ? window.claude.use('downloads').then(x => (dlCap = x)).catch(() => (dlCap = null)) : Promise.resolve(undefined);
  const dlmsg = $('#dlmsg');
  async function offer(zipName, files) {
    await dlReady;
    if (window.claude && window.claude.use) {
      if (!dlCap) { dlmsg.textContent = 'Downloads aren’t available in this view. Open the page in a browser where you’re signed in.'; return; }
      try {
        await dlCap.save({ filename: zipName, data: zip(files) });
        dlmsg.textContent = 'Saved ' + zipName + '.';
      } catch (err) {
        const code = err && err.code;
        dlmsg.textContent = code === 'declined' ? 'Download cancelled.' : code === 'rate_limited' ? 'Another download prompt is open. Finish it, then try again.' : 'That download didn’t go through (' + (code || 'error') + ').';
      }
      return;
    }
    // opened as a local file: save plain STLs
    files.forEach(f => {
      const blob = new Blob([f.data], { type: f.name.endsWith('.stl') ? 'model/stl' : 'text/plain' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = f.name; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    });
    dlmsg.textContent = 'Saved ' + files.length + ' file' + (files.length > 1 ? 's' : '') + '.';
  }
  const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const fileFor = r => {
    const p = r.parts[0];
    const nm = r.parts.length > 1 ? rowName(r) : p.name;
    return { name: 'heliotrope-' + slug(nm) + '.stl', data: stlBytes(p) };
  };

  const GROUPS = ['Viewer body', 'Optics', 'Eyepieces', 'Lid', 'Ornament', 'Mount', 'Pedestal', 'Drawer', 'Cards'];
  function buildPartsList() {
    const host = $('#parts');
    host.innerHTML = '';
    GROUPS.forEach(g => {
      const list = [...rows.values()].filter(r => r.parts[0].group === g);
      if (!list.length) return;
      const h = document.createElement('div'); h.className = 'grp'; h.textContent = g; host.appendChild(h);
      list.forEach(r => {
        const p = r.parts[0];
        let size = '';
        if (p._shells) { const pf = printFrame(p._shells, p.print); size = pf.size.map(v => v.toFixed(v < 10 ? 1 : 0)).join(' × ') + ' mm'; }
        const el = document.createElement('div');
        el.className = 'row'; el.tabIndex = 0; el.setAttribute('role', 'button');
        const count = r.parts.length > 1 ? r.parts.length : (p.qty || 1);
        const right = p.noExport ? `<span class="tag">${p.buy ? 'you supply' : '—'}</span>` : `<button class="dl" aria-label="Download ${esc(rowName(r))} STL">STL</button>`;
        el.innerHTML = `<span class="sw" style="background:${SWATCH[p.mat]}"></span>
          <div><div class="nm">${esc(rowName(r))}${count > 1 ? ` <span class="meta">×${count}</span>` : ''}</div>
          <div class="meta">${MATNAME[p.mat]}${size ? ' · ' + size : ''}${p.buy && !p.noExport ? ' · cut/buy, template' : ''}</div>
          ${p.note ? `<div class="note">${esc(p.note)}</div>` : ''}</div>${right}`;
        el.addEventListener('click', e => {
          if (e.target.classList.contains('dl')) {
            e.stopPropagation();
            const f = fileFor(r);
            offer(f.name.replace(/\.stl$/, '.zip'), [f]);
            return;
          }
          select(selKey === r.key ? null : r.key);
        });
        el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(selKey === r.key ? null : r.key); } });
        r.el = el; host.appendChild(el);
      });
    });
  }
  $('#dl-all').addEventListener('click', async e => {
    const btn = e.currentTarget; btn.disabled = true; const t = btn.textContent; btn.textContent = 'Preparing…';
    await new Promise(r => setTimeout(r, 30));
    const files = [...rows.values()].filter(r => !r.parts[0].noExport).map(fileFor);
    const lines = ['The Heliotrope stereoscope - STL files (millimetres, Z up, laid out for printing).', '',
      ...[...rows.values()].filter(r => !r.parts[0].noExport).map(r => {
        const n = r.parts.length > 1 ? r.parts.length : (r.parts[0].qty || 1);
        return `${'heliotrope-' + slug(rowName(r)) + '.stl'}  x${n}  [${MATNAME[r.parts[0].mat]}]  ${r.parts[0].note || ''}`;
      })];
    files.push({ name: 'README.txt', data: new TextEncoder().encode(lines.join('\r\n')) });
    await offer('heliotrope-stereoscope-stl.zip', files);
    btn.disabled = false; btn.textContent = t;
  });

  // ───────────────────────────── loop ──────────────────────────────────────
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  resize();
  function frame(now) {
    stepTween(now);
    if (mode === 'mount' && meshes.length) {
      if (!mountPaused) mountClock = (now - mountT0);
      mountPose(mountClock / 1000);
    }
    if (meshes.length) apply();
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  build().then(() => { setMode('assembled'); });
  requestAnimationFrame(frame);
})();
