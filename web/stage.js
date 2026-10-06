// The 3D stage: the fly band, the instruments, the lights. No network or audio logic here.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export const MEMBERS = {
  drums:  { name: 'Drums',    color: '#ff7a3d', pos: [0, .7, -3.1], rot: 0 },
  bass:   { name: 'Bass',     color: '#8f6bff', pos: [2.6, 0, 1.4], rot: -.35 },
  guitar: { name: 'Guitar',   color: '#ffd23f', pos: [-2.6, 0, 1.4], rot: .35 },
  synth:  { name: 'Synth',    color: '#3dd6c6', pos: [-3.9, 0, -1.6], rot: .55 },
  vocals: { name: 'Vocals',   color: '#ff4f9a', pos: [.9, 0, 2.6], rot: -.1 },
};
export const ORDER = ['drums', 'bass', 'guitar', 'synth', 'vocals'];  // order of the brains on the server

// ---------- small helpers ----------
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const std = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: .5, ...o });
const mesh = (geo, mat, [x, y, z] = [0, 0, 0]) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); return m; };
const sphere = (r, mat, pos, [sx, sy, sz] = [1, 1, 1]) => { const m = mesh(new THREE.SphereGeometry(r, 24, 16), mat, pos); m.scale.set(sx, sy, sz); return m; };
const cyl = (rt, rb, h, mat, pos, seg = 24) => mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat, pos);
const box = (w, h, d, mat, pos) => mesh(new THREE.BoxGeometry(w, h, d), mat, pos);
function segGeo(a, b, r, rb = r * .75, radial = 6) {  // geometry of a cylinder from a to b
  a = V(...a); b = V(...b);
  const d = b.clone().sub(a), g = new THREE.CylinderGeometry(r, rb, d.length(), radial);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), d.clone().normalize()));
  return g.translate(...a.addScaledVector(d, .5).toArray());
}
const seg = (a, b, r, mat) => new THREE.Mesh(segGeo(a, b, r), mat);
const boxGeo = (w, h, d, [x, y, z]) => new THREE.BoxGeometry(w, h, d).translate(x, y, z);
function canvasTex(w, h, draw, srgb = true) {
  const c = Object.assign(document.createElement('canvas'), { width: w, height: h });
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

// ---------- procedural textures (shared) ----------
const eyeTex = canvasTex(256, 256, (g, w, h) => {  // compound eye: hexagonal ommatidia
  g.fillStyle = '#3a0306'; g.fillRect(0, 0, w, h);
  const r = 5.5, dx = r * Math.sqrt(3), dy = r * 1.5;
  for (let row = 0, y = 0; y < h + r; row++, y += dy) for (let x = (row % 2) * dx / 2; x < w + r; x += dx) {
    g.beginPath();
    for (let k = 0; k < 6; k++) { const a = Math.PI / 6 + k * Math.PI / 3; g.lineTo(x + Math.cos(a) * r * .92, y + Math.sin(a) * r * .92); }
    const gr = g.createRadialGradient(x - 1.5, y - 1.5, .5, x, y, r);
    gr.addColorStop(0, '#ff6a5a'); gr.addColorStop(.45, '#c4141c'); gr.addColorStop(1, '#5a0408');
    g.fillStyle = gr; g.fill();
  }
});
const wingTex = canvasTex(256, 256, (g, w, h) => {  // transparent membrane with veins
  g.clearRect(0, 0, w, h);
  g.fillStyle = 'rgba(215,232,255,.32)'; g.beginPath(); g.arc(w / 2, h / 2, w / 2, 0, 7); g.fill();
  g.strokeStyle = 'rgba(40,28,22,.85)'; g.lineCap = 'round';
  for (const [y0, y1, lw] of [[.5, .5, 3], [.38, .3, 2], [.62, .7, 2], [.44, .4, 1.6], [.56, .6, 1.6], [.3, .2, 1.4], [.7, .8, 1.4]]) {
    g.lineWidth = lw; g.beginPath(); g.moveTo(w * .04, h * y0);
    g.bezierCurveTo(w * .35, h * (y0 + y1) / 2, w * .7, h * y1, w * .98, h * y1); g.stroke();
  }
  g.lineWidth = 1.4;
  for (const [x, a, b] of [[.45, .38, .56], [.62, .3, .44], [.7, .6, .7]]) { g.beginPath(); g.moveTo(w * x, h * a); g.lineTo(w * x + 6, h * b); g.stroke(); }
});
const glowSprite = canvasTex(64, 64, (g) => {
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(.4, 'rgba(255,255,255,.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
});

// ---------- Kevin Parker hair ----------
// 168 long wavy strands over a continuous veil, center parting, brown with sun-bleached tips.
// Merged into 6 clumps (left/right x front/side/back) that sway with different phases: 8 draw calls in total.
function makeHair() {
  const group = new THREE.Group(), clumps = [];
  const root = new THREE.Color('#6b4a2e'), tip = new THREE.Color('#c99a64'), tmp = new THREE.Color();
  const geos = Array.from({ length: 6 }, () => []);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const strand = (s, phi, R, L, ph, amp, top = .01) => {
    const sp = Math.sin(phi), cp = Math.cos(phi);
    const pts = [
      V(s * top, 1, .62 * cp).normalize().multiplyScalar(.312),                 // on the center parting
      V(s * sp * .75, .85, cp * .75).normalize().multiplyScalar(.35),           // puffed over the crown
      V(s * sp * R * .96, .1, cp * R * .96),
    ];
    for (let j = 1; j <= 6; j++) {  // the wavy fall, toward shoulders and chest
      const rr = R + .022 * j + amp * Math.sin(j * 1.25 + ph) + (j === 6 ? .03 : 0);
      const a = phi + amp * .9 * Math.cos(j * 1.1 + ph);  // the waves also wrap around the head
      pts.push(V(s * Math.sin(a) * rr, .1 - j * (L + .1) / 6, Math.cos(a) * rr - (phi < 1.5 ? -.02 : .028) * j));  // the front strands fall onto the chest
    }
    return new THREE.CatmullRomCurve3(pts);
  };
  const shade = (t, sun, k) => tmp.copy(root).lerp(tip, THREE.MathUtils.smoothstep(t, .4, 1) * sun).multiplyScalar(k);
  const N = 168, locks = Array.from({ length: 40 }, () => ({ ph: rnd() * 6.28, amp: .018 + rnd() * .02, L: .78 + rnd() * .16 }));
  for (let i = 0; i < N; i++) {
    const s = i % 2 ? 1 : -1, k = (Math.floor(i / 2) + rnd() * .8) / (N / 2);
    const phi = THREE.MathUtils.degToRad(62 + 118 * k);  // 0 = face, 180 = nape
    const lock = locks[Math.floor(k * 19.99) * 2 + (s > 0 ? 1 : 0)];  // neighboring strands sway together
    const R = (phi < 1.75 ? .45 : phi < 2.1 ? .42 : .37) + rnd() * .025;           // wide at the sides to keep the eyes clear
    const L = lock.L + rnd() * .06, ph = lock.ph + rnd() * .5, amp = lock.amp;
    const tubular = 36, radial = 4;
    const g = new THREE.TubeGeometry(strand(s, phi, R, L, ph, amp, .01 + rnd() * .012), tubular, .0095 + rnd() * .006, radial, false);
    const n = g.attributes.position.count, col = new Float32Array(n * 3), dim = .8 + rnd() * .35, sun = .65 + rnd() * .35;
    for (let v = 0; v < n; v++) {
      const t = Math.floor(v / (radial + 1)) / tubular;
      shade(t, sun, dim);
      col.set([tmp.r, tmp.g, tmp.b], v * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const band = phi < 1.75 ? 0 : phi < 2.5 ? 1 : 2;
    geos[band * 2 + (s > 0 ? 1 : 0)].push(g);
  }
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .42, metalness: .08, emissive: 0x2a1608 });
  geos.forEach((list, c) => {
    const m = new THREE.Mesh(mergeGeometries(list), mat);
    m.castShadow = true; m.userData = { side: c % 2 ? 1 : -1, phase: c * 1.3, weight: 1 + Math.floor(c / 2) * .3 };
    group.add(m); clumps.push(m);
  });
  for (const s of [-1, 1]) {  // continuous veil under the strands: gives the hair mass and silhouette
    const NU = 44, NV = 40, pos = [], col = [], idx = [];
    for (let u = 0; u <= NU; u++) {
      const phi = THREE.MathUtils.degToRad(64 + 116 * u / NU);
      const c = strand(s, phi, (phi < 1.75 ? .435 : phi < 2.1 ? .41 : .36), .8 + .06 * Math.sin(u * 1.7), phi * 5, .02, .006);
      for (let v = 0; v <= NV; v++) { pos.push(...c.getPoint(v / NV).toArray()); shade(v / NV, .75, .7); col.push(tmp.r, tmp.g, tmp.b); }
    }
    for (let u = 0; u < NU; u++) for (let v = 0; v < NV; v++) {
      const a = u * (NV + 1) + v, b = a + NV + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    const veil = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .5, side: THREE.DoubleSide, emissive: 0x1e1006 }));
    veil.castShadow = true; clumps[2 + (s > 0 ? 1 : 0)].add(veil);
  }
  const cap = mesh(new THREE.SphereGeometry(.318, 28, 14, 0, Math.PI * 2, 0, 1.3), new THREE.MeshStandardMaterial({ color: 0x4e3120, roughness: .6 }));
  cap.rotation.x = -.25; group.add(cap);
  return { group, clumps };
}

// ---------- the singer's look (from the reference photos in ref/) ----------
// cream mohair sweater with brown Fair Isle bands, white mirrored sport sunglasses, necklaces.
function fairIsle(bands) {  // pattern bands on cream wool; bands = [[y0, y1] as fractions of the height]
  return canvasTex(512, 256, (g, w, h) => {
    g.fillStyle = '#eee4d2'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(${Math.random() < .5 ? '255,250,240' : '190,170,140'},.35)`; g.fillRect(Math.random() * w, Math.random() * h, 2, 1); }
    for (const [a, b] of bands) {
      const y0 = a * h, y1 = b * h, hh = y1 - y0, n = 16, cw = w / n;
      g.fillStyle = '#4a2e1c'; g.fillRect(0, y0, w, hh * .1); g.fillRect(0, y1 - hh * .1, w, hh * .1);
      for (let i = 0; i < n; i++) {  // alternating leaves / diamonds
        const cx = (i + .5) * cw, cy = (y0 + y1) / 2;
        g.fillStyle = i % 2 ? '#4a2e1c' : '#7a5a3c';
        g.beginPath(); g.moveTo(cx, y0 + hh * .18); g.quadraticCurveTo(cx + cw * .45, cy, cx, y1 - hh * .18);
        g.quadraticCurveTo(cx - cw * .45, cy, cx, y0 + hh * .18); g.fill();
        g.fillStyle = '#4a2e1c'; g.fillRect(cx + cw * .5 - 2, cy - 2, 4, 4);
      }
    }
  });
}
const speckle = canvasTex(256, 256, (g, w, h) => {  // mohair fuzz: speckles for alphaMap
  g.fillStyle = '#000'; g.fillRect(0, 0, w, h); g.fillStyle = '#fff';
  for (let i = 0; i < 5000; i++) g.fillRect(Math.random() * w, Math.random() * h, 1 + Math.random() * 2, 1);
}, false);
export const shiny = [];  // materials that get the envMap (mirrored lenses, glossy plastic) in createStage
function dressKevin(body, head, arms) {
  const wool = (map) => std(0xffffff, { map, bumpMap: speckle, bumpScale: 1.5, roughness: 1 });
  const fuzz = new THREE.MeshStandardMaterial({ color: 0xf2eadb, alphaMap: speckle, alphaTest: .5, roughness: 1 });
  const chest = wool(fairIsle([[.42, .62]]));
  const thorax = sphere(.435, chest, [0, 1.58, 0], [1.05, .9, 1]);
  const hem = mesh(new THREE.SphereGeometry(.445, 32, 12, 0, Math.PI * 2, 0, 1.15), wool(fairIsle([[.72, .92]])), [0, .9, -.05]);
  hem.scale.set(1, 1.35, .95);
  body.add(thorax, hem);
  for (const m of [thorax, hem]) { const f = m.clone(); f.material = fuzz; f.scale.multiplyScalar(1.03); f.userData.noShadow = true; body.add(f); }
  const cuffMat = wool(fairIsle([[0, 1]]));
  arms.forEach((p, i) => {  // sleeves on the front legs, with a patterned cuff
    const s = i ? 1 : -1;
    p.add(new THREE.Mesh(segGeo([0, .02, -.02], [s * .1, -.25, .3], .1, .085, 14), chest));
    p.add(new THREE.Mesh(segGeo([s * .085, -.21, .25], [s * .1, -.25, .3], .095, .095, 14), cuffMat));
  });
  // wraparound sport sunglasses: glossy white frame, iridescent mirrored lenses
  const frame = new THREE.MeshPhysicalMaterial({ color: 0xd9d5cc, roughness: .3, clearcoat: 1, clearcoatRoughness: .15 });
  const lens = new THREE.MeshPhysicalMaterial({ color: 0x8fa3bf, metalness: 1, roughness: .1, iridescence: 1,
    iridescenceIOR: 1.8, iridescenceThicknessRange: [200, 600], transparent: true, opacity: .9 });
  shiny.push(frame, lens);
  for (const s of [-1, 1]) {
    const n = V(s * .78, .08, .62).normalize(), g = new THREE.Group();
    g.position.set(s * .22, .04, .08);
    g.add(mesh(new THREE.SphereGeometry(.215, 28, 10, 0, Math.PI * 2, 0, 1.05), lens));
    const rim = mesh(new THREE.TorusGeometry(.215 * Math.sin(1.05), .026, 8, 36), frame, [0, .215 * Math.cos(1.05), 0]);
    rim.rotation.x = Math.PI / 2; g.add(rim);
    g.scale.set(.86, 1, 1.05);
    g.quaternion.setFromUnitVectors(V(0, 1, 0), n);
    head.add(g);
    head.add(new THREE.Mesh(segGeo([s * .4, .06, -.02], [s * .3, .05, -.24], .022, .016, 8), frame));  // temple arm
  }
  head.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([V(-.2, .09, .27), V(0, .13, .33), V(.2, .09, .27)]), 12, .022, 8), frame));
  // necklaces: two gold chains and one of beads, layered on the chest
  const gold = std(0xe0b050, { metalness: 1, roughness: .25 });
  const neck = d => new THREE.CatmullRomCurve3([V(-.3, 1.86, -.05), V(-.33, 1.68, .26), V(-.16, 1.45 - d, .45 + d * .3),
    V(0, 1.4 - d, .47 + d * .3), V(.16, 1.45 - d, .45 + d * .3), V(.33, 1.68, .26), V(.3, 1.86, -.05)]);
  body.add(new THREE.Mesh(mergeGeometries([new THREE.TubeGeometry(neck(0), 48, .009, 5), new THREE.TubeGeometry(neck(.2), 48, .007, 5)]), gold));
  const beads = [], curve = neck(.1), palette = ['#3fb7a8', '#f0e6d0', '#7a4a2a', '#d98a3a'].map(c => new THREE.Color(c));
  for (let i = 0; i <= 34; i++) {
    const b = new THREE.SphereGeometry(.022, 8, 6).translate(...curve.getPoint(i / 34).toArray()), c = palette[i % 4];
    b.setAttribute('color', new THREE.Float32BufferAttribute(Array(b.attributes.position.count).fill(0).flatMap(() => [c.r, c.g, c.b]), 3));
    beads.push(b);
  }
  body.add(new THREE.Mesh(mergeGeometries(beads), std(0xffffff, { vertexColors: true, roughness: .35 })));
}

// ---------- fly ----------
function makeFly(color, key) {
  const g = new THREE.Group();
  const chitin = std(0x3b2b1e, { roughness: .4, metalness: .25 });
  const tan = std(0x8d6a3d, { roughness: .6 });
  const leg = std(0x22180f, { roughness: .7 });
  const eyeMat = std(0xffffff, { map: eyeTex, roughness: .25, metalness: .1, emissive: 0x300000 });
  const antMat = std(0x5a4430, { emissive: new THREE.Color(color), emissiveIntensity: 0 });
  const body = new THREE.Group(); g.add(body);
  body.add(sphere(.42, tan, [0, .9, -.05], [1, 1.35, .95]));
  for (const y of [.62, .85, 1.08]) {  // abdomen stripes
    const t = mesh(new THREE.TorusGeometry(.4 - Math.abs(y - .85) * .5, .035, 8, 32), chitin, [0, y, -.05]);
    t.rotation.x = Math.PI / 2; t.scale.z = .95; body.add(t);
  }
  body.add(sphere(.4, chitin, [0, 1.58, 0], [1.05, .9, 1]));
  const head = new THREE.Group(); head.position.set(0, 2.06, .06); body.add(head);
  head.add(sphere(.3, chitin));
  head.add(sphere(.2, eyeMat, [-.22, .04, .08], [.8, 1, .9]), sphere(.2, eyeMat, [.22, .04, .08], [.8, 1, .9]));
  head.add(seg([0, -.12, .22], [0, -.32, .34], .035, leg));  // proboscis
  head.add(seg([-.07, .16, .26], [-.13, .34, .4], .03, antMat), seg([.07, .16, .26], [.13, .34, .4], .03, antMat));  // antennae = ears
  // bristles: a few, on the thorax and the back of the head, all in a single geometry
  const bristles = [];
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2, top = i % 2;
    const base = top ? V(Math.cos(a) * .25, 1.86, Math.sin(a) * .2 - .05) : V(Math.cos(a) * .36, 1.62, Math.sin(a) * .32);
    const dir = base.clone().sub(V(0, 1.45, 0)).normalize();
    bristles.push(segGeo(base.toArray(), base.clone().addScaledVector(dir, .16 + (i % 3) * .04).toArray(), .012, .002, 4));
  }
  body.add(new THREE.Mesh(mergeGeometries(bristles), leg));
  const wingMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, map: wingTex, transparent: true, roughness: .15,
    iridescence: 1, iridescenceIOR: 1.6, side: THREE.DoubleSide, depthWrite: false });
  const wings = [-1, 1].map(s => {
    const pivot = new THREE.Group(); pivot.position.set(s * .15, 1.72, -.32);
    const w = mesh(new THREE.CircleGeometry(1, 32), wingMat, [s * .55, -.35, 0]);
    w.scale.set(.62, .22, 1); w.rotation.z = s * -.6; w.userData.noShadow = true;
    pivot.add(w); body.add(pivot); return pivot;
  });
  for (const s of [-1, 1]) {  // hind and middle legs
    body.add(seg([s * .25, .75, .05], [s * .45, .35, .25], .04, leg), seg([s * .45, .35, .25], [s * .4, 0, .4], .035, leg));
    body.add(seg([s * .36, 1.42, .05], [s * .72, 1.15, .15], .04, leg), seg([s * .72, 1.15, .15], [s * .8, .75, .25], .035, leg));
  }
  const arms = [-1, 1].map(s => {  // front legs = arms
    const p = new THREE.Group(); p.position.set(s * .32, 1.64, .22);
    p.add(seg([0, 0, 0], [s * .1, -.25, .3], .045, leg), seg([s * .1, -.25, .3], [s * .1, -.45, .62], .04, leg));
    body.add(p); return p;
  });
  const hair = key === 'vocals' ? makeHair() : null;
  if (hair) { head.add(hair.group); dressKevin(body, head, arms); }
  return { g, body, head, wings, arms, antMat, hair, hairLag: 0, hairVel: 0 };
}

// ---------- instruments ----------
const glow = color => std(color, { roughness: .3, metalness: .4, emissive: new THREE.Color(color), emissiveIntensity: 0 });
const metal = std(0x9a9aa8, { metalness: .9, roughness: .25 });
const GUITAR_OUTLINE = [[0, -.46], [.17, -.44], [.27, -.34], [.29, -.2], [.24, -.07], [.21, .02], [.25, .13], [.27, .25], [.22, .33],
  [.15, .3], [.11, .2], [.07, .15], [-.07, .15], [-.11, .19], [-.17, .24], [-.21, .21], [-.22, .1], [-.2, .01], [-.25, -.1],
  [-.29, -.24], [-.26, -.36], [-.16, -.44]];
function makeStringed(big, color) {
  const inst = new THREE.Group(), bodyMat = glow(color);
  const sx = big ? 1.08 : 1, sy = big ? 1.15 : 1;
  const shape = new THREE.Shape();
  const P = GUITAR_OUTLINE.map(([x, y]) => new THREE.Vector2(x * sx, y * sy));
  shape.moveTo(P[0].x, P[0].y); shape.splineThru([...P.slice(1), P[0]]);
  const bodyGeo = new THREE.ExtrudeGeometry(shape, { depth: .07, bevelEnabled: true, bevelThickness: .022, bevelSize: .02,
    bevelSegments: 3, curveSegments: 12 }).translate(0, -.1, -.035);
  inst.add(new THREE.Mesh(bodyGeo, bodyMat));
  const len = big ? 1.3 : .95, y0 = .06, nutY = y0 + len, w = big ? .085 : .072;
  const wood = std(0x3a2414, { roughness: .7 }), dark = std(0x16100c, { roughness: .45 });
  inst.add(box(w, len, .05, wood, [0, y0 + len / 2, .005]));                 // neck
  inst.add(box(w * .95, len, .02, dark, [0, y0 + len / 2, .04]));           // fretboard
  const S = len * 1.45, frets = [];
  for (let k = 1; k <= 20; k++) frets.push(boxGeo(w * .95, .006, .012, [0, nutY - S * (1 - 2 ** (-k / 12)), .052]));
  inst.add(new THREE.Mesh(mergeGeometries(frets), metal));
  const hs = new THREE.Shape();  // headstock
  hs.moveTo(-w / 2, 0); hs.lineTo(w / 2, 0); hs.lineTo(w * .7, .12); hs.quadraticCurveTo(w * .9, .3, .02, .33);
  hs.quadraticCurveTo(-w * .7, .3, -w * .65, .12); hs.closePath();
  const head = new THREE.Mesh(new THREE.ExtrudeGeometry(hs, { depth: .035, bevelEnabled: true, bevelThickness: .008, bevelSize: .008, bevelSegments: 2 }), bodyMat);
  head.position.set(0, nutY, -.01); head.rotation.x = -.18; inst.add(head);
  const nStr = big ? 4 : 6, pegs = [];
  for (let i = 0; i < nStr; i++) pegs.push(new THREE.CylinderGeometry(.012, .012, .05, 8).rotateZ(Math.PI / 2)
    .translate((i % 2 ? 1 : -1) * w * .62, nutY + .07 + Math.floor(i / 2) * (big ? .1 : .07), .015));
  inst.add(new THREE.Mesh(mergeGeometries(pegs), metal));
  const pick = std(0xf3efe6, { roughness: .35 });  // pickups, bridge, knobs
  for (const y of big ? [-.12] : [-.06, -.22]) inst.add(box(w * 1.15, .05, .02, dark, [0, y, .062]));
  inst.add(box(w * 1.2, .035, .02, metal, [0, -.33, .062]));
  for (const [x, y] of [[.16, -.3], [.2, -.38], [.11, -.4]]) { const k = cyl(.02, .022, .03, pick, [x * sx, y * sy, .065], 12); k.rotation.x = Math.PI / 2; inst.add(k); }
  const strings = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .3 });
  const sg = [];
  for (let i = 0; i < nStr; i++) sg.push(boxGeo(big ? .005 : .003, nutY + .33, .003, [(i - (nStr - 1) / 2) * w * .75 / nStr, (nutY - .33) / 2, .068]));
  inst.add(new THREE.Mesh(mergeGeometries(sg), strings));
  // correct left-handed hold: the neck points to the fly's left (+x)
  inst.position.set(big ? -.12 : -.1, 1.15, .58); inst.rotation.set(-.15, big ? -.25 : .25, -1.0);
  return { inst, strings, bodyMat };
}
function makeAmp(color) {
  const g = new THREE.Group(), tolex = std(0x15111a, { roughness: .85 });
  const grille = canvasTex(128, 128, (c, w, h) => {
    c.fillStyle = '#2a2420'; c.fillRect(0, 0, w, h);
    c.strokeStyle = 'rgba(0,0,0,.45)'; c.lineWidth = 1;
    for (let i = -h; i < w; i += 4) { c.beginPath(); c.moveTo(i, 0); c.lineTo(i + h, h); c.stroke(); c.beginPath(); c.moveTo(i + h, 0); c.lineTo(i, h); c.stroke(); }
  });
  g.add(box(.82, .82, .42, tolex, [0, .41, 0]));
  g.add(mesh(new THREE.PlaneGeometry(.72, .72), std(0xffffff, { map: grille, roughness: .9 }), [0, .41, .212]));
  g.add(box(.82, .26, .4, tolex, [0, .95, -.01]));
  const panel = std(0xcfc6b8, { metalness: .6, roughness: .35 });
  g.add(box(.74, .12, .01, panel, [0, .95, .195]));
  const knobs = [];
  for (let i = 0; i < 6; i++) knobs.push(new THREE.CylinderGeometry(.018, .02, .03, 10).rotateX(Math.PI / 2).translate(-.26 + i * .09, .95, .21));
  g.add(new THREE.Mesh(mergeGeometries(knobs), std(0x111111)));
  const lamp = glow(color);
  g.add(box(.74, .025, .012, lamp, [0, 1.06, .195]));
  return { g, lamp };
}
function makeInstrument(key, color) {
  const g = new THREE.Group(), glows = [], parts = {};
  if (key === 'drums') {
    const shell = glow(color), skin = std(0xf2ead8, { roughness: .8, emissive: new THREE.Color(color), emissiveIntensity: 0 });
    const brass = std(0xd8a838, { metalness: .9, roughness: .3, emissive: new THREE.Color(0xffc040), emissiveIntensity: 0 });
    parts.kick = new THREE.Group(); parts.kick.position.set(0, .5, .95);
    const k = cyl(.5, .5, .42, shell); k.rotation.x = Math.PI / 2; parts.kick.add(k);
    const kf = cyl(.47, .47, .43, skin); kf.rotation.x = Math.PI / 2; kf.scale.y = .99; parts.kick.add(kf);
    parts.snare = new THREE.Group(); parts.snare.position.set(-.55, 1.0, .75);
    parts.snare.add(cyl(.28, .28, .16, shell), cyl(.27, .27, .17, skin)); parts.snare.rotation.x = .25;
    const tom = cyl(.22, .22, .2, shell, [.35, 1.25, .75]); tom.rotation.x = .4;
    parts.hat = new THREE.Group(); parts.hat.position.set(-.95, 1.3, .55);
    parts.hat.add(cyl(.3, .3, .02, brass), cyl(.3, .3, .02, brass, [0, .05, 0]));
    const ride = cyl(.42, .42, .02, brass, [.9, 1.55, .5]); ride.rotation.x = .35;
    g.add(parts.kick, parts.snare, tom, parts.hat, ride,
      seg([-.95, 0, .55], [-.95, 1.3, .55], .02, metal), seg([.9, 0, .5], [.9, 1.55, .5], .02, metal),
      cyl(.25, .25, .5, std(0x2a2030), [0, .25, -.15]));  // stool
    glows.push(shell, skin, brass);
  } else if (key === 'bass' || key === 'guitar') {
    const s = makeStringed(key === 'bass', color), amp = makeAmp(color);
    amp.g.position.set(key === 'bass' ? .75 : -.75, 0, -1.05);
    g.add(s.inst, amp.g); parts.inst = s.inst; parts.strings = s.strings; glows.push(s.bodyMat, amp.lamp);
  } else if (key === 'synth') {
    const caseMat = glow(color);
    g.add(box(1.7, .14, .55, caseMat, [0, 1.12, .78]));
    g.add(seg([-.6, 0, .6], [.6, 1.05, .9], .025, metal), seg([.6, 0, .6], [-.6, 1.05, .9], .025, metal));
    parts.keys = [];
    for (let i = 0; i < 24; i++) {
      const black = [1, 3, 6, 8, 10].includes(i % 12);
      const m = std(black ? 0x111111 : 0xf2efe8, { emissive: new THREE.Color(color), emissiveIntensity: 0 });
      const k = box(black ? .04 : .062, .05, black ? .2 : .32, m, [-.78 + i * .068, black ? 1.23 : 1.2, black ? .7 : .78]);
      k.userData.y = k.position.y; parts.keys.push(k); g.add(k);
    }
    glows.push(caseMat);
  } else if (key === 'vocals') {
    const mic = glow(color);
    g.add(seg([0, 0, .8], [0, 1.66, .64], .025, metal), cyl(.22, .22, .03, metal, [0, .015, .8]));
    parts.mic = sphere(.08, mic, [0, 1.77, .6]);  // below the mouth
    g.add(parts.mic, seg([0, 1.66, .64], [0, 1.72, .61], .04, std(0x222222)));
    glows.push(mic);
  }
  return { g, glows, parts };
}

// ---------- volumetric light cones (fake: additive, faded) ----------
const coneVert = `varying float vY; varying vec3 vN, vV;
  void main(){ vY = uv.y; vec4 mv = modelViewMatrix * vec4(position,1.); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }`;
const coneFrag = `uniform vec3 uColor; uniform float uOpacity; varying float vY; varying vec3 vN, vV;
  void main(){ float facing = abs(dot(normalize(vN), normalize(vV)));
    float a = uOpacity * pow(vY, 1.6) * pow(facing, 1.8) * smoothstep(1., .96, vY);
    gl_FragColor = vec4(uColor * a, 1.); }`;
function makeCone(from, to, angle, color) {
  const d = to.clone().sub(from), L = d.length();
  const geo = new THREE.ConeGeometry(Math.tan(angle) * L * .85, L, 40, 1, true).translate(0, -L / 2, 0);
  const mat = new THREE.ShaderMaterial({ uniforms: { uColor: { value: new THREE.Color(color) }, uOpacity: { value: .1 } },
    vertexShader: coneVert, fragmentShader: coneFrag, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  const m = new THREE.Mesh(geo, mat);
  m.position.copy(from); m.quaternion.setFromUnitVectors(V(0, -1, 0), d.normalize());
  return m;
}

// ---------- truss ----------
function trussGeo(a, b, w = .26) {  // square-section beam: 4 chords + diagonals
  a = V(...a); b = V(...b);
  const d = b.clone().sub(a), L = d.length(), parts = [];
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, z]) => [x * w / 2, z * w / 2]);
  for (const [x, z] of corners) parts.push(segGeo([x, 0, z], [x, L, z], .022, .022, 6));
  const n = Math.max(2, Math.round(L / w));
  for (let i = 0; i < n; i++) for (let c = 0; c < 4; c++) {
    const [x0, z0] = corners[c], [x1, z1] = corners[(c + 1) % 4];
    parts.push(segGeo([x0, i * L / n, z0], [x1, (i + 1) * L / n, z1], .009, .009, 4));
  }
  const g = mergeGeometries(parts);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), d.normalize()));
  return g.translate(a.x, a.y, a.z);
}

// =====================================================================
export function createStage(parent) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  parent.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x07050d, 16, 34);
  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, .1, 100);
  const HOME = { pos: V(0, 5.6, 14.2), target: V(0, 1.7, -.4) };
  camera.position.copy(HOME.pos);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(HOME.target);
  controls.enableDamping = true;
  controls.maxPolarAngle = 1.45;
  controls.minDistance = 2.5; controls.maxDistance = 22;
  let goal = null;  // camera move in progress
  controls.addEventListener('start', () => { goal = null; });

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), .55, .45, .82);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  let frame = null;  // free rectangle (not covered by the UI) where the band is centered
  function applyFrame() {
    const W = innerWidth, H = innerHeight;
    camera.aspect = W / H;
    camera.fov = Math.max(42, 2 * THREE.MathUtils.radToDeg(Math.atan(Math.tan(THREE.MathUtils.degToRad(18)) / camera.aspect)));  // narrow screens: fit the whole band in width
    if (frame) {  // move the optical center to the center of the free area
      const cx = frame.left + frame.width / 2, cy = frame.top + frame.height / 2;
      camera.setViewOffset(W, H, W / 2 - cx, H / 2 - cy, W, H);
    } else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  }
  let size = '';
  function fit() {  // checked every frame: catches "silent" resizes too (headless, browser bars)
    const W = innerWidth, H = innerHeight;
    if (size === W + 'x' + H) return;
    size = W + 'x' + H;
    renderer.setSize(W, H); composer.setSize(W, H); bloom.setSize(W, H); applyFrame();
  }
  applyFrame();

  // lights
  scene.add(new THREE.HemisphereLight(0x8070b0, 0x100818, .55));
  const key = new THREE.DirectionalLight(0xffe2c0, 1.1);
  key.position.set(3, 9, 8); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 1, far: 30 });
  key.shadow.bias = -.0004; key.shadow.normalBias = .02;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x9a6bff, .6); rim.position.set(-4, 5, -8); scene.add(rim);

  // stage
  const floor = cyl(7.6, 8, .4, std(0x2a2138, { roughness: .55, metalness: 0 }), [0, -.2, 0], 72);
  floor.receiveShadow = true; scene.add(floor);
  const edge = mesh(new THREE.TorusGeometry(7.65, .04, 8, 120), new THREE.MeshBasicMaterial({ color: 0xff4f9a }));
  edge.rotation.x = Math.PI / 2; scene.add(edge);
  const riser = box(3.4, .7, 2.4, std(0x1d1628, { roughness: .4 }), [0, .35, -3.2]);  // drum riser
  riser.receiveShadow = riser.castShadow = true; scene.add(riser);

  // "Currents"-style backdrop: waves that light up when the band plays well
  const backdrop = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uEnergy: { value: .3 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }',
    fragmentShader: `uniform float uTime, uEnergy; varying vec2 vUv;
      void main(){
        vec2 p = vUv * 2. - 1.; p.x *= 2.2; p.y += .35;
        float r = length(p), a = atan(p.y, p.x);
        float w = sin(11. * r - uTime * 1.3 + 2.2 * sin(a * 3. + uTime * .25));
        vec3 col = mix(vec3(.30,.07,.48), vec3(.98,.40,.16), smoothstep(-.3, .95, w));
        col = mix(col, vec3(.03,.02,.06), smoothstep(.5, 1.9, r));
        col = pow(col * (.18 + .82 * uEnergy), vec3(2.2)) * 1.25;  // sRGB to linear: OutputPass handles the rest
        gl_FragColor = vec4(col, 1.);
      }`,
  });
  scene.add(mesh(new THREE.PlaneGeometry(26, 11), backdrop, [0, 4.6, -6.5]));

  // lighting truss: a rectangle suspended on four towers
  const TY = 6.3, TX = 5.2, TZ0 = -1.4, TZ1 = 4.4;
  const truss = mergeGeometries([
    trussGeo([-TX, TY, TZ0], [TX, TY, TZ0]), trussGeo([-TX, TY, TZ1], [TX, TY, TZ1]),
    trussGeo([-TX, TY, TZ0], [-TX, TY, TZ1]), trussGeo([TX, TY, TZ0], [TX, TY, TZ1]),
    ...[[-TX, TZ0], [TX, TZ0], [-TX, TZ1], [TX, TZ1]].map(([x, z]) => trussGeo([x, 0, z], [x, TY, z])),
  ]);
  scene.add(new THREE.Mesh(truss, std(0x8b8a96, { metalness: .85, roughness: .35 })));

  // haze: slowly drifting particles
  const HAZE = 420, hazePos = new Float32Array(HAZE * 3), hazeSeed = new Float32Array(HAZE);
  for (let i = 0; i < HAZE; i++) {
    hazePos.set([(Math.random() - .5) * 16, Math.random() * 6.5, -5 + Math.random() * 11], i * 3); hazeSeed[i] = Math.random() * 100;
  }
  const hazeGeo = new THREE.BufferGeometry(); hazeGeo.setAttribute('position', new THREE.BufferAttribute(hazePos, 3));
  const haze = new THREE.Points(hazeGeo, new THREE.PointsMaterial({ map: glowSprite, size: .9, color: 0x6a4a9a, transparent: true,
    opacity: .09, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(haze);

  // sound rings (one wave per played note)
  const rings = Array.from({ length: 60 }, () => {
    const m = mesh(new THREE.TorusGeometry(.25, .018, 6, 40),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    m.life = 0; m.visible = false; scene.add(m); return m;
  });
  let ringNext = 0;
  function ring(pos, color) {
    const r = rings[ringNext++ % rings.length];
    r.position.copy(pos); r.scale.setScalar(1); r.life = 1; r.visible = true; r.material.color.set(color);
    r.lookAt(camera.position);
  }

  // on-stage monitor of neural activity
  const monCanvas = Object.assign(document.createElement('canvas'), { width: 320, height: 160 });
  const mon = monCanvas.getContext('2d');
  mon.fillStyle = '#05030a'; mon.fillRect(0, 0, 320, 160);
  const monTex = new THREE.CanvasTexture(monCanvas); monTex.colorSpace = THREE.SRGBColorSpace;
  const monitor = new THREE.Group();
  monitor.position.set(3.9, 0, -1.8); monitor.rotation.y = -.55;
  monitor.add(box(1.9, 1.05, .08, std(0x0d0a14, { metalness: .5 }), [0, 1.9, 0]));
  monitor.add(mesh(new THREE.PlaneGeometry(1.8, .95), new THREE.MeshBasicMaterial({ map: monTex, color: 0xb0b0b0 }), [0, 1.9, .045]));
  monitor.add(seg([0, 0, 0], [0, 1.4, 0], .04, std(0x333344, { metalness: .8 })));
  scene.add(monitor);

  // band assembly
  const band = {};
  for (const [k, m] of Object.entries(MEMBERS)) {
    const fly = makeFly(m.color, k), inst = makeInstrument(k, m.color);
    const root = new THREE.Group();
    root.position.set(...m.pos); root.rotation.y = m.rot;
    root.add(fly.g, inst.g);
    if (k === 'drums') fly.g.position.set(0, .28, -.15);  // sitting on the stool
    root.traverse(o => { if (o.isMesh && !o.userData.noShadow) o.castShadow = true; });
    const from = V(THREE.MathUtils.clamp(m.pos[0] * .8, -TX + .5, TX - .5), TY - .25, m.pos[2] + 2.5 > 1.5 ? TZ1 : TZ0);
    const spot = new THREE.SpotLight(new THREE.Color(m.color).lerp(new THREE.Color(0xfff0e0), .45), 30, 14, .36, .6, 1.4);  // more neutral light: true colors stay readable
    spot.position.copy(from); spot.target = root;
    const can = cyl(.13, .17, .32, std(0x111111, { metalness: .6, roughness: .4 }), from.toArray());
    can.quaternion.setFromUnitVectors(V(0, -1, 0), V(...m.pos).sub(from).normalize());
    const cone = makeCone(from.clone().add(V(0, -.15, 0)), V(m.pos[0], m.pos[1], m.pos[2]), .36, m.color);
    scene.add(root, spot, can, cone);
    band[k] = { ...m, key: k, fly, inst, root, spot, cone, hit: 0, arm: 0, act: 0 };
  }

  const env = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), .04).texture;  // reflections for lenses and plastic
  for (const m of shiny) { m.envMap = env; m.envMapIntensity = m.metalness ? .6 : .3; }

  // ---------- API ----------
  let beat = 0, energy = .3, focused = null;
  const at = V(0, 0, 0);
  function perform(k, notes) {
    const m = band[k];
    if (!m || !notes.length) return;
    m.hit = 1; m.arm ^= 1;
    const p = m.inst.parts;
    if (k === 'drums') for (const [c] of notes) {
      if (!p[c]) continue;
      p[c].userData.hit = 1; p[c].getWorldPosition(at); ring(at, m.color);
    } else {
      if (p.keys) for (const [c] of notes) { const key = p.keys[((c - 48) % 24 + 24) % 24]; key.userData.hit = 1; }
      (p.inst || p.mic || m.root).getWorldPosition(at);
      if (!p.inst && !p.mic) at.y += 1.2;
      if (!p.mic) ring(at, m.color);  // vocals already have the mic lighting up: no ring on the face
    }
  }
  function focus(k, instant = false) {  // instant: no camera flight (direct links, screenshots)
    focused = band[k] ? k : null;
    if (!focused) { goal = { pos: HOME.pos.clone(), target: HOME.target.clone() }; return; }
    const m = band[k], fwd = V(Math.sin(m.rot - .25), 0, Math.cos(m.rot - .25));  // a bit of three-quarter view
    const target = V(...m.pos).add(V(0, k === 'drums' ? 1.6 : 1.5, 0));
    goal = { target, pos: target.clone().addScaledVector(fwd, 5.4).add(V(0, 1.1, 0)) };
    if (instant) { camera.position.copy(goal.pos); controls.target.copy(goal.target); goal = null; }
  }
  function monitorColumn(levels, played) {  // one column per sixteenth, one row per brain
    mon.drawImage(monCanvas, -3, 0);
    ORDER.forEach((k, i) => {
      mon.fillStyle = '#05030a'; mon.fillRect(317, i * 32, 3, 32);
      mon.globalAlpha = Math.min(1, levels[i]); mon.fillStyle = MEMBERS[k].color; mon.fillRect(317, i * 32 + 2, 3, 28);
      mon.globalAlpha = 1;
      if (played[i]) { mon.fillStyle = '#fff'; mon.fillRect(317, i * 32 + 14, 3, 4); }
    });
    monTex.needsUpdate = true;
  }

  // ---------- animation ----------
  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    fit();
    const dt = Math.min(clock.getDelta(), .05), t = clock.elapsedTime;
    backdrop.uniforms.uTime.value = t;
    backdrop.uniforms.uEnergy.value += (energy * (.75 + .25 * beat) - backdrop.uniforms.uEnergy.value) * .08;
    beat *= Math.exp(-dt * 6);
    for (const m of Object.values(band)) {
      const { fly, inst } = m;
      m.hit *= Math.exp(-dt * 9);
      fly.body.position.y = m.hit * .07 + beat * .03;
      const nod = m.key === 'vocals' ? .26 : .18;
      fly.head.rotation.x = -.1 + beat * nod + m.hit * .1;
      fly.head.rotation.z = m.key === 'vocals' ? Math.sin(t * 1.1) * .06 : 0;
      fly.arms.forEach((a, i) => a.rotation.x = -(i === m.arm ? m.hit : m.hit * .3) * .7);
      const flap = (14 + 46 * m.act) * t;  // wingbeat = brain activity
      fly.wings.forEach((w, i) => w.rotation.y = (i ? -1 : 1) * (.25 + .25 * Math.sin(flap)));
      fly.antMat.emissiveIntensity *= Math.exp(-dt * 10);
      if (fly.hair) {  // the hair follows the head on a damped spring: lag + bounce
        const hx = fly.head.rotation.x;
        fly.hairVel += ((hx - fly.hairLag) * 60 - fly.hairVel * 9) * dt;
        fly.hairLag += fly.hairVel * dt;
        const swing = (fly.hairLag - hx) * 1.1;
        for (const c of fly.hair.clumps) {
          const { side, phase, weight } = c.userData;
          c.rotation.x = swing * weight + Math.sin(t * 1.3 + phase) * .025;
          c.rotation.z = side * (.02 + Math.sin(t * .9 + phase) * .025 + m.hit * .03) - fly.head.rotation.z * .6;
        }
      }
      for (const g of inst.glows) g.emissiveIntensity = m.hit * .7;
      m.spot.intensity = 18 + 40 * m.hit;
      m.cone.material.uniforms.uOpacity.value = (focused && focused !== m.key ? .03 : .07) + m.hit * .12;
      const p = inst.parts;
      for (const o of [p.kick, p.snare, p.hat, ...(p.keys || [])]) {
        if (!o) continue;
        o.userData.hit = (o.userData.hit || 0) * Math.exp(-dt * 12);
        if (o === p.kick) o.scale.setScalar(1 + o.userData.hit * .08);
        else if (o === p.hat) o.rotation.z = Math.sin(t * 40) * o.userData.hit * .12;
        else if (o === p.snare) o.position.y = 1.0 - o.userData.hit * .03;
        else { o.material.emissiveIntensity = o.userData.hit * 1.5; o.position.y = o.userData.y - o.userData.hit * .025; }
      }
      if (p.inst) { p.inst.rotation.x = -.15 + m.hit * .12; p.strings.opacity = .3 + m.hit * .7; }
      if (p.mic) p.mic.scale.setScalar(1 + m.hit * .25);
    }
    for (const r of rings) if (r.life > 0) {
      r.life -= dt * 1.5; r.scale.setScalar(1 + (1 - r.life) * 2.6); r.material.opacity = Math.max(0, r.life) * .45;
      if (r.life <= 0) r.visible = false;
    }
    for (let i = 0; i < HAZE; i++) {  // slow drift, wraps around to the other side
      const s = hazeSeed[i];
      hazePos[i * 3] += Math.sin(t * .1 + s) * dt * .15 + dt * .06;
      hazePos[i * 3 + 1] += Math.cos(t * .13 + s) * dt * .05;
      if (hazePos[i * 3] > 8) hazePos[i * 3] = -8;
    }
    hazeGeo.attributes.position.needsUpdate = true;
    haze.material.opacity = .07 + beat * .04;
    edge.material.color.setHSL((t * .03) % 1, .9, .5 + beat * .2);
    if (goal) {
      const a = 1 - Math.exp(-dt * 3.2);
      camera.position.lerp(goal.pos, a); controls.target.lerp(goal.target, a);
      if (camera.position.distanceTo(goal.pos) < .01) goal = null;
    }
    controls.update();
    composer.render();
  });

  return {
    perform,
    focus,
    monitorColumn,
    setActivity(k, v) { if (band[k]) band[k].act = Math.max(0, Math.min(1, v)); },
    setEnergy(x) { energy = x; },
    onBeat() { beat = 1; },
    hearStep() { for (const m of Object.values(band)) m.fly.antMat.emissiveIntensity = 1.2; },
    setFrame(rect) { frame = rect; applyFrame(); },
    setUIHidden(hidden) { if (hidden) { frame = null; applyFrame(); } },
  };
}
