// Validates the prototype kit: every printed shell closed and consistently oriented. node check-kit.js
globalThis.THREE = require('three');
require('./geometry.js'); require('./proto.js');
const t0 = Date.now();
(async () => {
const kit = await Helio.buildKit(Helio.buildParts(), (i, n, k) => { if (process.env.V) console.log('…', k.id, Date.now() - t0); });
let bad = 0;
for (const p of kit.parts) {
  const msg = p._shells.map((s, si) => {
    const P = s.p, E = new Map(), k = i => Math.round(P[i] * 1e3) + ',' + Math.round(P[i + 1] * 1e3) + ',' + Math.round(P[i + 2] * 1e3);
    for (let i = 0; i < P.length; i += 9) { const a = k(i), b = k(i + 3), c = k(i + 6); if (a === b || b === c || a === c) continue;
      for (const [u, v] of [[a, b], [b, c], [c, a]]) E.set(u + '>' + v, (E.get(u + '>' + v) || 0) + 1); }
    let be = 0; for (const [key, n] of E) { const [u, v] = key.split('>'); if (n !== 1 || E.get(v + '>' + u) !== 1) be++; }
    if (be) bad++;
    return `s${si}: ${P.length / 9} tris, ${(s.volume() / 1000).toFixed(1)} cm³${be ? ', BAD EDGES ' + be : ''}`;
  });
  console.log(p.id.padEnd(18), 'x' + (p.qty || 1), msg.join(' | '));
}
console.log('hardware:', kit.hardware.map(h => h.n + ' × ' + h.label).join('; '));
console.log('fasteners', kit.fasteners.length, 'bad shells', bad, 'ms', Date.now() - t0);
})();
