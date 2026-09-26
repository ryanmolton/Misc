// Validates that every part is a closed, consistently oriented mesh: npm i three@0.147.0 && node check-geometry.js
globalThis.THREE = require('three');
require('./geometry.js');
const t0=Date.now();
const parts = Helio.buildParts();
let total=0;
for (const p of parts) {
  const shells = p.shells();
  let msg=[];
  shells.forEach((s,si)=>{
    const P=s.p; const E=new Map(); const k=i=>Math.round(P[i]*1e4)+','+Math.round(P[i+1]*1e4)+','+Math.round(P[i+2]*1e4);
    let degenerate=0;
    for(let i=0;i<P.length;i+=9){const a=k(i),b=k(i+3),c=k(i+6); if(a===b||b===c||a===c){degenerate++;continue;}
      for(const [u,v] of [[a,b],[b,c],[c,a]]){const key=u+'>'+v;E.set(key,(E.get(key)||0)+1);}}
    let bad=0; for(const [key,n] of E){const [u,v]=key.split('>'); if(n!==1||E.get(v+'>'+u)!==1) bad++;}
    msg.push(`shell${si}: tris=${P.length/9} vol=${(s.volume()/1000).toFixed(1)}cm3 badEdges=${bad} degen=${degenerate}`);
    total+=P.length/9;
  });
  console.log(p.id.padEnd(16), msg.join(' | '));
}
console.log('total tris',total,'ms',Date.now()-t0);
