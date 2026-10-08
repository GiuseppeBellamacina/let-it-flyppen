// The 3D stage: the fly band, the instruments, the lights. No network or audio logic here.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

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
// fluffy white mohair sweater with charcoal Nordic rosettes, ivory sunglasses with dark tinted lenses, beaded necklaces.
function fairIsle(bands) {  // pattern bands on white mohair; bands = [[y0, y1] as fractions of the height]
  return canvasTex(512, 256, (g, w, h) => {
    g.fillStyle = '#ece8df'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 4000; i++) {  // fuzzy fibres
      g.strokeStyle = `rgba(${Math.random() < .6 ? '255,253,248' : '196,188,174'},.4)`; g.lineWidth = 1;
      const x = Math.random() * w, y = Math.random() * h, a = Math.random() * 6.3, l = 2 + Math.random() * 5;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke();
    }
    g.filter = 'blur(1.6px)';  // mohair: the knit motifs are soft-edged, never crisp
    for (const [a, b] of bands) {
      const y0 = a * h, y1 = b * h, hh = y1 - y0, n = 8, cw = w / n, cy = (y0 + y1) / 2, ink = '#2c2a28';
      g.fillStyle = ink;
      for (let x = 4; x < w; x += 12) {  // dotted rows above and below the band
        g.beginPath(); g.arc(x, y0 + hh * .06, hh * .045, 0, 7); g.arc(x + 6, y1 - hh * .06, hh * .045, 0, 7); g.fill();
      }
      for (let i = 0; i < n; i++) {
        const cx = (i + .5) * cw;
        if (i % 2 === 0) {  // big rosette: five rounded petals round a dark core
          for (let p = 0; p < 5; p++) {
            const t = p / 5 * Math.PI * 2 - Math.PI / 2;
            g.beginPath(); g.ellipse(cx + Math.cos(t) * hh * .22, cy + Math.sin(t) * hh * .22, hh * .2, hh * .12, t, 0, 7); g.fill();
          }
          g.beginPath(); g.arc(cx, cy, hh * .14, 0, 7); g.fill();
        } else {  // pair of leaves on a stem
          g.beginPath(); g.ellipse(cx - cw * .14, cy, cw * .2, hh * .15, -.5, 0, 7); g.ellipse(cx + cw * .14, cy, cw * .2, hh * .15, .5, 0, 7); g.fill();
          g.fillRect(cx - 1.5, cy - hh * .3, 3, hh * .6);
        }
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
  const fly = { g, body, head, wings, arms, antMat, hair, hairLag: 0, hairVel: 0 };
  fly.animate = (m, t, dt, beat) => {
    body.position.y = m.hit * .07 + beat * .03;
    head.rotation.x = -.1 + beat * (key === 'vocals' ? .26 : .18) + m.hit * .1;
    head.rotation.z = key === 'vocals' ? Math.sin(t * 1.1) * .06 : 0;
    arms.forEach((a, i) => a.rotation.x = -(i === m.arm ? m.hit : m.hit * .3) * .7);
    const flap = (14 + 46 * m.act) * t;  // wingbeat = brain activity
    wings.forEach((w, i) => w.rotation.y = (i ? -1 : 1) * (.25 + .25 * Math.sin(flap)));
    antMat.emissiveIntensity *= Math.exp(-dt * 10);
    if (hair) swayHair(fly, m, t, dt, head.rotation.x, head.rotation.z);
  };
  return fly;
}
function swayHair(fly, m, t, dt, hx, hz = 0) {  // the hair follows the head on a damped spring: lag + bounce
  fly.hairVel += ((hx - fly.hairLag) * 60 - fly.hairVel * 9) * dt;
  fly.hairLag += fly.hairVel * dt;
  const swing = (fly.hairLag - hx) * 1.1;
  for (const c of fly.hair.clumps) {
    const { side, phase, weight } = c.userData;
    c.rotation.x = swing * weight + Math.sin(t * 1.3 + phase) * .025;
    c.rotation.z = side * (.02 + Math.sin(t * .9 + phase) * .025 + m.hit * .03) - hz * .6;
  }
}

// ---------- flybody fly (data/fly.glb, made by `prep.py fly` from the Janelia/DeepMind MuJoCo model) ----------
// glTF nodes = MuJoCo bodies with their hinge joints in userData.joints; MuJoCo frame: x forward, y left, z up.
// The musician stands up: head up (+Y), belly to the audience (+Z), its left on +X.
const FLY_H = 2.3;
const ORIENT = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(V(0, 1, 0), V(1, 0, 0), V(0, 0, -1)));
const _jm = new THREE.Matrix4(), _jq = new THREE.Quaternion(), _jv = V(0, 0, 0), _jp = V(0, 0, 0);
function rig(model) {  // q[joint] = angle; apply(): node.matrix = rest * J1 * J2 ..., J = T(pos) R(axis, q) T(-pos), XML order
  const q = {}, def = {}, nodes = {}, bodies = [];
  model.traverse(o => {
    nodes[o.name] = o;
    if (o.isMesh || !o.userData.joints) return;
    o.updateMatrix(); o.matrixAutoUpdate = false;
    bodies.push({ o, rest: o.matrix.clone(), joints: o.userData.joints });
    for (const j of o.userData.joints) { q[j.name] = j.springref; def[j.name] = j; }
  });
  function apply() {
    for (const { o, rest, joints } of bodies) {
      o.matrix.copy(rest);
      for (const j of joints) {
        _jq.setFromAxisAngle(_jv.fromArray(j.axis), q[j.name]);
        _jm.makeRotationFromQuaternion(_jq).setPosition(_jp.fromArray(j.pos).sub(_jv.fromArray(j.pos).applyQuaternion(_jq)));
        o.matrix.multiply(_jm);
      }
      o.matrixWorldNeedsUpdate = true;
    }
  }
  return { q, def, nodes, apply };
}
// coordinate descent IK: move `names` joints so that node `eff` reaches `target` (in the space of `root`, detached)
function reach(r, root, names, eff, target, pull = .004) {
  const p = V(0, 0, 0);
  const lim = names.map(n => { const [a, b] = r.def[n].range; return a === b ? [-3, 3] : [a - .25, b + .25]; });
  const cost = () => {
    r.apply(); root.updateMatrixWorld(true);
    return eff.getWorldPosition(p).distanceToSquared(target) + pull * names.reduce((s, n) => s + (r.q[n] - r.def[n].springref) ** 2, 0);
  };
  let best = cost();
  for (let step = .3; step > .004;) {
    let moved = false;
    names.forEach((n, i) => {
      for (const d of [step, -step]) {
        const old = r.q[n]; r.q[n] = THREE.MathUtils.clamp(old + d, ...lim[i]);
        const c = cost();
        if (c < best - 1e-7) { best = c; moved = true; break; }
        r.q[n] = old;
      }
    });
    if (!moved) step *= .5;
  }
  r.apply(); root.updateMatrixWorld(true);
  return Math.sqrt(best);
}
const legJoints = (r, leg, side, upTo = 'tarsus') => ['coxa', 'femur', 'tibia', 'tarsus'].slice(0, ['coxa', 'femur', 'tibia', 'tarsus'].indexOf(upTo) + 1)
  .flatMap(b => r.nodes[`${b}_${leg}_${side}`].userData.joints.map(j => j.name));
const SIDES = { left: 1, right: -1 };  // the fly's left is +X on stage

function flyMaterials(proto) {  // once, on the shared prototype: amber cuticle, red eyes, glassy wings
  const done = new Map();
  proto.traverse(o => {
    if (!o.isMesh) return;
    const name = o.material.name;
    if (!done.has(name)) {
      const c = o.material.color.clone().convertSRGBToLinear();  // MuJoCo rgba are display colors
      done.set(name,
        name === 'body' ? new THREE.MeshPhysicalMaterial({ color: c, roughness: .42, clearcoat: .6, clearcoatRoughness: .35, sheen: .4, sheenColor: 0xffb070 }) :
        name === 'lower' ? new THREE.MeshPhysicalMaterial({ color: c, roughness: .55, sheen: .3, sheenColor: 0xffe0b0 }) :
        name === 'red' ? new THREE.MeshPhysicalMaterial({ color: 0xffffff, map: eyeTex, roughness: .3, clearcoat: 1, clearcoatRoughness: .2, emissive: 0x2a0000 }) :
        name === 'membrane' ? new THREE.MeshPhysicalMaterial({ color: 0xdfe8f5, transparent: true, opacity: .32, roughness: .12, iridescence: 1,
          iridescenceIOR: 1.6, iridescenceThicknessRange: [250, 700], side: THREE.DoubleSide, depthWrite: false }) :
        name === 'brown' ? new THREE.MeshStandardMaterial({ color: 0x3a2214, roughness: .5, side: THREE.DoubleSide }) :
        name === 'ocelli' ? new THREE.MeshPhysicalMaterial({ color: 0x5a2a10, roughness: .15, clearcoat: 1 }) :
        new THREE.MeshStandardMaterial({ color: 0x120c08, roughness: .5 }));  // black / bristle-brown: bristles
    }
    o.material = done.get(name);
    if (name === 'red' && !o.geometry.attributes.uv) {  // ommatidia: planar UVs across each eye (seen from the side)
      const p = o.geometry.attributes.position, uv = new Float32Array(p.count * 2);
      o.geometry.computeBoundingBox();
      const bb = o.geometry.boundingBox, size = bb.getSize(V(0, 0, 0)), axes = [0, 1, 2].sort((a, b) => size.getComponent(a) - size.getComponent(b));
      const [u, v] = [axes[1], axes[2]], k = 30 / size.getComponent(v);  // ~30 facets along the eye
      for (let i = 0; i < p.count; i++) { uv[i * 2] = p.getComponent(i, u) * k; uv[i * 2 + 1] = p.getComponent(i, v) * k; }
      o.geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    }
    if (o.material.transparent || /black|brown|tarsus|claw/.test(o.name)) o.userData.noShadow = true;
  });
  eyeTex.wrapS = eyeTex.wrapT = THREE.RepeatWrapping; eyeTex.needsUpdate = true;
}

// arm targets (where the front-leg tips go), in the member's root space
const HANDS = {
  drums:  { left: [.38, 1.32, .72], right: [-.52, 1.08, .72] },   // tom / snare
  bass:   { left: [.42, 1.38, .62], right: [-.08, 1.02, .7] },    // neck / strings
  guitar: { left: [.4, 1.36, .62], right: [-.08, 1.0, .7] },
  synth:  { left: [.32, 1.26, .74], right: [-.32, 1.26, .74] },   // keys
  vocals: { left: [.32, 1.55, .66], right: [-.06, 1.6, .64] },    // gesture / mic stand
};
function makeFlybody(proto, color, key, offset) {
  const g = new THREE.Group(), body = new THREE.Group(), orient = new THREE.Group(), model = proto.clone(true);
  g.position.copy(offset); orient.quaternion.copy(ORIENT);
  g.add(body); body.add(orient); orient.add(model);
  const r = rig(model), n = r.nodes, q = r.q;
  const antMat = new THREE.MeshPhysicalMaterial({ color: 0x9a6a3a, roughness: .45, emissive: new THREE.Color(color).multiplyScalar(.15), emissiveIntensity: 0 });
  for (const s of ['left', 'right']) n[`antenna_${s}-body`].material = antMat;
  const setup = () => { r.apply(); g.updateMatrixWorld(true); };  // g is detached here: world = the member's root space
  // head: pitch it so the face looks at the audience
  const pitch = n.head.userData.joints.map(j => j.name).find(j => j === 'head');
  const face = V(0, 0, 0), hp = V(0, 0, 0), want = V(0, .25, 1).normalize();
  let bestA = 0, bestD = -2;
  for (let a = -1.6; a <= 1.6; a += .05) {
    q[pitch] = a; setup();
    n.head.getWorldPosition(hp);
    face.addVectors(n.antenna_left.getWorldPosition(V(0, 0, 0)), n.antenna_right.getWorldPosition(V(0, 0, 0))).multiplyScalar(.5).sub(hp).normalize();
    if (face.dot(want) > bestD) { bestD = face.dot(want); bestA = a; }
  }
  q[pitch] = bestA; setup();
  // scale: thorax + head + abdomen = most of the height, the legs lift the abdomen tip a little off the floor
  const box = new THREE.Box3();
  for (const name of ['thorax-body', 'head-body', 'head-red', 'abdomen_8-body', 'abdomen_7-body']) if (n[name]) box.expandByObject(n[name], true);
  const s = (FLY_H - .15) / (box.max.y - box.min.y);
  orient.scale.setScalar(s); setup();
  box.makeEmpty();
  for (const name of ['thorax-body', 'head-body', 'head-red', 'abdomen_8-body', 'abdomen_7-body']) if (n[name]) box.expandByObject(n[name], true);
  orient.position.y = .15 + offset.y - box.min.y; setup();
  // standing on the hind and middle legs, front legs to the instrument
  const feet = { T3: [.42, 0, .2], T2: [.72, 0, .62] };
  for (const [side, sx] of Object.entries(SIDES)) {
    for (const [leg, [x, y, z]] of Object.entries(feet)) reach(r, g, legJoints(r, leg, side), n[`claw_${leg}_${side}`], V(sx * x, y, z));
    const [x, y, z] = HANDS[key][side];
    reach(r, g, legJoints(r, 'T1', side), n[`claw_T1_${side}`], V(x, y, z));
  }
  const base = { ...q };
  const fly = { g, body, head: n.head, wings: [], arms: [], antMat, hair: null, hairLag: 0, hairVel: 0, rig: r, base, model, n, s };
  fly.animate = (m, t, dt, beat) => {
    Object.assign(q, base);
    body.position.y = m.hit * .02 + beat * .01;
    q[pitch] = base[pitch] + beat * (key === 'vocals' ? .16 : .1) + m.hit * .06;
    q.head_twist = Math.sin(t * 1.1 + ORDER.indexOf(key) * 1.7) * (key === 'vocals' ? .12 : .05);
    for (const [side] of Object.entries(SIDES)) {  // strokes: the playing arm swings at the femur-tibia joint
      const h = (side === 'left') === (m.arm === 1) ? m.hit : m.hit * .3;
      q[`tibia_T1_${side}`] += h * .55; q[`femur_T1_${side}`] -= h * .25;
      const buzz = m.act * (.5 + .5 * Math.sin((14 + 46 * m.act) * t));  // wings: open and buzz with brain activity
      q[`wing_yaw_${side}`] = base[`wing_yaw_${side}`] - .35 * m.act - .25 * buzz;
      q[`wing_roll_${side}`] = base[`wing_roll_${side}`] + .3 * buzz;
    }
    for (let i = 2; i <= 7; i++) q[`abdomen_${i}`] += Math.sin(t * 2.2 - i * .4) * .015;  // breathing
    q.abdomen += Math.sin(t * 2.2) * .015;
    r.apply();
    antMat.emissiveIntensity *= Math.exp(-dt * 10);
    if (fly.hair) swayHair(fly, m, t, dt, q[pitch] - base[pitch]);
  };
  return fly;
}

// ---------- Kevin Parker on the flybody fly: everything measured on the posed meshes (fly space = fly.g) ----------
const strandTex = canvasTex(64, 256, (g, w, h) => {  // hair card: wavy strands of different lengths on transparent
  g.clearRect(0, 0, w, h);
  for (let i = 0; i < 70; i++) {
    const x0 = Math.random() * w, len = h * (.7 + Math.random() * .3), l = 180 + Math.random() * 75;
    g.strokeStyle = `rgba(${l},${l},${l},${.55 + Math.random() * .45})`; g.lineWidth = .8 + Math.random() * 1.8;
    g.beginPath(); g.moveTo(x0, 0);
    for (let y = 8; y <= len; y += 8) g.lineTo(x0 + Math.sin(y * .05 + i) * 2.5, y);
    g.stroke();
  }
});
function meshBox(meshes, into = new THREE.Box3(), filter = () => true) {  // precise box in fly space (g detached at the origin offset)
  const v = V(0, 0, 0);
  for (const m of meshes) {
    const p = m.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(m.matrixWorld); if (filter(v)) into.expandByPoint(v); }
  }
  return into;
}
function anchorAt(node, flyWorld, center) {  // group under `node` whose frame is fly space moved to `center` (follows the node)
  const a = new THREE.Group(); a.matrixAutoUpdate = false;
  a.matrix.copy(node.matrixWorld).invert().multiply(flyWorld).multiply(new THREE.Matrix4().makeTranslation(center.x, center.y, center.z));
  node.add(a); return a;
}
function shell(mesh, d, flyWorld, uvOf) {  // copy of a mesh pushed out along its normals by d (fly units), UVs from fly-space position
  const geo = mesh.geometry.clone(), p = geo.attributes.position, nrm = geo.attributes.normal, uv = new Float32Array(p.count * 2);
  const toFly = new THREE.Matrix4().copy(flyWorld).invert().multiply(mesh.matrixWorld), k = d / mesh.matrixWorld.getMaxScaleOnAxis(), v = V(0, 0, 0);
  for (let i = 0; i < p.count; i++) {
    p.setXYZ(i, p.getX(i) + nrm.getX(i) * k, p.getY(i) + nrm.getY(i) * k, p.getZ(i) + nrm.getZ(i) * k);
    uv.set(uvOf(v.fromBufferAttribute(p, i).applyMatrix4(toFly)), i * 2);
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geo;
}
function wrapGeo(pts, margin, NT = 48, NY = 30) {  // smooth tube round a point cloud (radius per height and angle), cylindrical UVs
  const b = new THREE.Box3().setFromPoints(pts), c = b.getCenter(V(0, 0, 0)), y0 = b.min.y, y1 = b.max.y;
  let r = Array.from({ length: NY + 1 }, () => new Array(NT).fill(0));
  for (const p of pts) {
    const iy = Math.round((p.y - y0) / (y1 - y0) * NY), it = Math.floor(((Math.atan2(p.x - c.x, p.z - c.z) / (2 * Math.PI) + 1) % 1) * NT) % NT;
    r[iy][it] = Math.max(r[iy][it], Math.hypot(p.x - c.x, p.z - c.z));
  }
  for (let pass = 0; pass < 16; pass++) r = r.map((row, y) => row.map((v, t) => {  // fill the gaps, then smooth: no steps between rows
    let sum = 0, k = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dt = -2; dt <= 2; dt++) { const w = r[THREE.MathUtils.clamp(y + dy, 0, NY)][(t + dt + NT) % NT]; if (w) { sum += w; k++; } }
    return pass < 4 && v ? v : k ? sum / k : 0;
  }));
  const rows = r.map((row, y) => [y0 + (y1 - y0) * y / NY, row.map(v => v + margin)]);
  rows.push([y1 + margin * .4, r[NY].map(v => v * .75 + margin)], [y1 + margin * .6, r[NY].map(v => v * .55)]);  // shoulders and crew neck
  const pos = [], uv = [], idx = [], R = rows.length;
  rows.forEach(([y, row], iy) => { for (let t = 0; t <= NT; t++) {
    const a = t / NT * Math.PI * 2, rr = row[t % NT];
    pos.push(c.x + Math.sin(a) * rr, y, c.z + Math.cos(a) * rr); uv.push(t / NT * 2, (y - y0) / (y1 - y0));  // two pattern repeats round the body
  } });
  for (let iy = 0; iy < R - 1; iy++) for (let t = 0; t < NT; t++) { const a = iy * (NT + 1) + t, d = a + NT + 1; idx.push(a, a + 1, d, d, a + 1, d + 1); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx); geo.computeVertexNormals();
  return geo;
}
function dressKevinFly(fly) {
  const { n, g } = fly, W = g.matrixWorld, off = g.position, local = b => b.translate(off.clone().negate());
  const headBox = local(meshBox([n['head-body'], n['head-red']])), hc = headBox.getCenter(V(0, 0, 0)), hs = headBox.getSize(V(0, 0, 0)).multiplyScalar(.5);
  const eyes = [1, -1].map(s => local(meshBox([n['head-red']], new THREE.Box3(), v => (v.x - off.x - hc.x) * s > 0)));  // the fly's left eye is on +X
  const tbox = local(meshBox([n['thorax-body']]));
  const head = anchorAt(n.head, W, hc);  // head frame: origin at the head center, fly axes, stage units

  // --- hair cards: centre part, volume on the sides, falling behind the eyes to the shoulders ---
  const hair = new THREE.Group(), clumps = [], geos = Array.from({ length: 6 }, () => []);
  const root = new THREE.Color('#24170e'), tip = new THREE.Color('#5e3e25'), tmp = new THREE.Color();  // dark brown shag, slightly warmer ends
  const ex = Math.max(eyes[0].max.x, -eyes[1].min.x) - hc.x;  // half width over the eyes
  // lens surface of the sunglasses (built below): an ellipse around (0, ec.y, ec.z), from angle T0 at the nose to T1 at the temple
  const eb = eyes[0].clone().union(eyes[1]), ec = eb.getCenter(V(0, 0, 0)).sub(hc), eh = eb.getSize(V(0, 0, 0)).multiplyScalar(.5);
  const ax = ex * 1.12, az = eb.max.z - hc.z - ec.z + hs.z * .4, hh = eh.y * 1.05;
  const T0 = Math.asin(Math.min(.9, (eyes[0].min.x - hc.x) / ax)), T1 = 1.95;
  const keepOffGlasses = p => {  // at lens height the hair passes behind the lens edge and outside its surface, never through it
    const band = (p.y - ec.y) / (hh * 1.45);
    if (Math.abs(band) >= 1) return p;
    const k = (1 - band * band) ** 2, u = p.x / ax, w = (p.z - ec.z) / az;
    const r = Math.hypot(u, w), th = Math.atan2(Math.abs(u), w);
    if (th > T1 + .45) return p;  // well behind the temples: free
    const th2 = th + (Math.max(th, T1 + .12) - th) * k, r2 = r + (Math.max(r, 1.22) - r) * k, s = p.x < 0 ? -1 : 1;
    return p.set(s * Math.sin(th2) * r2 * ax, p.y, ec.z + Math.cos(th2) * r2 * az);
  };
  const drop = hs.y * 1.55;  // shag: from the temples down to the jaw / neck, not the shoulders
  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const strand = (s, phi, R, L, ph, amp, top = phi) => {  // top != phi: swept from the front of the parting round to the side
    const X = ex * R, Y = hs.y * 1.1, Z = hs.z * R, P = (a, r, y) => V(s * Math.sin(a) * X * r, y, Math.cos(a) * Z * r);
    const pts = [V(s * .015, Y, Math.cos(top) * Z * .75), P(top, .6, Y * .93), P((top + phi) / 2, .97, Y * (top < phi ? .72 : .45)), P(phi, 1, 0)];
    for (let j = 1; j <= 5; j++) {
      const a = phi + amp * Math.cos(j * 1.1 + ph), rr = 1 + .05 * j + amp * 2 * Math.sin(j * 1.3 + ph) + .24 * (j / 5) ** 2;  // ends flick outwards
      pts.push(V(s * Math.sin(a) * X * rr, -j * L / 5, Math.cos(a) * Z * rr - .02 * j));
    }
    return new THREE.CatmullRomCurve3(pts);
  };
  const NS = 24;
  const card = (c, width, dim, sun) => {  // a ribbon of hair along curve c, darker at the root
    const pos = [], col = [], uv = [], idx = [];
    for (let j = 0; j <= NS; j++) {
      const t = j / NS, p = c.getPoint(t), tg = c.getTangent(t), out = V(p.x, 0, p.z).normalize();
      const side = V(0, 0, 0).crossVectors(tg, out).normalize().multiplyScalar(width * (1 - .35 * t) / 2);
      pos.push(...keepOffGlasses(p.clone().sub(side)).toArray(), ...keepOffGlasses(p.clone().add(side)).toArray());
      tmp.copy(root).lerp(tip, THREE.MathUtils.smoothstep(t, .35, 1) * sun).multiplyScalar(dim);
      col.push(tmp.r, tmp.g, tmp.b, tmp.r, tmp.g, tmp.b); uv.push(0, 1 - t, 1, 1 - t);
      if (j < NS) idx.push(j * 2, j * 2 + 1, j * 2 + 2, j * 2 + 1, j * 2 + 3, j * 2 + 2);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); geo.setIndex(idx); geo.computeVertexNormals();
    return geo;
  };
  for (let layer = 0; layer < 4; layer++) for (let i = 0; i < (layer < 3 ? 24 : 14); i++) {
    const s = i % 2 ? 1 : -1, k = (Math.floor(i / 2) + rnd() * .6) / (layer < 3 ? 12 : 7), sweep = layer === 3;
    const phi = THREE.MathUtils.degToRad(sweep ? 84 + 20 * k : 60 + 120 * k + layer * 4);  // 0 = face, 180 = nape
    const top = sweep ? THREE.MathUtils.degToRad(12 + 50 * k) : phi;
    const R = (sweep ? 1.3 : 1.16 + layer * .08) + rnd() * .04, L = drop * (.75 + rnd() * .45 - layer * .05);  // choppy, uneven layers
    const c = strand(s, phi, R, L, rnd() * 6.3, .05 + rnd() * .05, top);
    const geo = card(c, (.22 + rnd() * .08) * (ex + hs.z) * (1 - layer * .1), (.78 + layer * .14) * (.85 + rnd() * .3), .6 + rnd() * .4);
    geos[(phi < 1.75 ? 0 : phi < 2.5 ? 1 : 2) * 2 + (s > 0 ? 1 : 0)].push(geo);
  }
  // shaggy fringe: from the crown forward over the forehead, ending just on the top of the frame
  const fringeEnd = ec.y + hh * 1.3;
  for (let i = 0; i < 18; i++) {
    const a = -.95 + 1.9 * (i + rnd() * .5) / 18, s = a < 0 ? -1 : 1, X = ex, Y = hs.y * 1.1, Z = hs.z;
    const P = (r, y, da = 0) => V(Math.sin(a + da) * X * r, y, Math.cos(a + da) * Z * r);
    const c = new THREE.CatmullRomCurve3([P(.3, Y * 1.02), P(.85, Y * .97), P(1.12, (Y + fringeEnd) / 2, a * .08),
      P(1.2, fringeEnd + rnd() * hh * .25, a * .18 + (rnd() - .5) * .1)]);
    geos[s > 0 ? 1 : 0].push(card(c, (.16 + rnd() * .06) * (ex + hs.z), .9 + rnd() * .2, .5 + rnd() * .3));
  }
  const hairMat = new THREE.MeshStandardMaterial({ vertexColors: true, map: strandTex, alphaTest: .35, side: THREE.DoubleSide, roughness: .5, emissive: 0x1a0e05 });
  geos.forEach((list, c) => {
    const m = new THREE.Mesh(mergeGeometries(list), hairMat);
    m.userData = { side: c % 2 ? 1 : -1, phase: c * 1.3, weight: 1 + Math.floor(c / 2) * .3 };
    hair.add(m); clumps.push(m);
  });
  const capEnd = Math.min(1.25, Math.acos(THREE.MathUtils.clamp((ec.y + hh * 1.25) / (hs.y * 1.08), -1, 1)));  // stops above the frame
  const cap = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, capEnd), new THREE.MeshStandardMaterial({ color: 0x24170e, roughness: .7 }));
  cap.scale.set(ex * 1.06, hs.y * 1.08, hs.z * 1.08); cap.position.z = -hs.z * .1; hair.add(cap);
  head.add(hair); fly.hair = { group: hair, clumps };

  // --- ivory wraparound sunglasses, dark lenses with a yellow-olive sheen, over the red eyes ---
  const frame = new THREE.MeshPhysicalMaterial({ color: 0xcfc4a4, roughness: .4, clearcoat: .6, clearcoatRoughness: .25 });  // ivory, not white: bloom picks up near-white
  const lens = new THREE.MeshPhysicalMaterial({ color: 0x3d3814, metalness: .55, roughness: .06, clearcoat: 1, iridescence: .35,
    iridescenceIOR: 1.8, iridescenceThicknessRange: [200, 600] });
  shiny.push(frame, lens);
  const lensPt = (s, th, yn) => {  // yn in [-1, 1] across the lens height; th from the nose round to the temple
    const f = Math.sqrt(1 - (.45 * yn) ** 2), shape = 1 - .3 * ((th - 1) / 1) ** 2;
    return V(s * Math.sin(th) * ax * f, ec.y + yn * hh * shape, ec.z + Math.cos(th) * az * f);
  };
  for (const s of [1, -1]) {
    const NU = 28, NV = 8, pos = [], idx = [];
    for (let u = 0; u <= NU; u++) for (let v = 0; v <= NV; v++) pos.push(...lensPt(s, T0 + (T1 - T0) * u / NU, v / NV * 2 - 1).toArray());
    for (let u = 0; u < NU; u++) for (let v = 0; v < NV; v++) { const a = u * (NV + 1) + v, b = a + NV + 1; idx.push(...(s > 0 ? [a, b, a + 1, b, b + 1, a + 1] : [a, a + 1, b, b, a + 1, b + 1])); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
    const lm = new THREE.Mesh(geo, lens); lm.userData.noShadow = true; head.add(lm);
    const rim = [];  // frame all round the lens, then the temple arm back into the hair
    for (let u = 0; u <= 16; u++) rim.push(lensPt(s, T0 + (T1 - T0) * u / 16, 1.05));
    for (let u = 16; u >= 0; u--) rim.push(lensPt(s, T0 + (T1 - T0) * u / 16, -1.05));
    const rimR = hh * .1;  // chunky frame
    head.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rim, true), 80, rimR, 8, true), frame));
    const end = lensPt(s, T1, .3);
    head.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([end, end.clone().add(V(s * .01, 0, -hs.z * .6)), V(s * ax * .9, ec.y, -hs.z * 1.05)]), 16, rimR * .8, 6), frame));
  }
  const a = lensPt(1, T0, .7), b = lensPt(-1, T0, .7);  // bridge over the face
  head.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([a, V(0, a.y + hh * .1, a.z + hh * .08), b]), 12, hh * .1, 8), frame));

  // --- cream mohair Fair-Isle sweater: a smooth shell shrink-wrapped round thorax + upper abdomen, sleeves on the front legs ---
  const torso = [n['thorax-body'], n['abdomen-body'], n['abdomen-lower'], n['abdomen_2-body'], n['abdomen_2-lower']].filter(Boolean);
  const pts = torso.flatMap(m => { const p = m.geometry.attributes.position, o = []; for (let i = 0; i < p.count; i += 2) o.push(V(0, 0, 0).fromBufferAttribute(p, i).applyMatrix4(m.matrixWorld).sub(off)); return o; });
  const knit = fairIsle([[.16, .36], [.46, .6], [.84, .98]]);  // canvas rows from the top: yoke, chest and hem bands
  knit.wrapS = THREE.RepeatWrapping;
  const wool = new THREE.MeshPhysicalMaterial({ color: 0xffffff, map: knit, bumpMap: speckle, bumpScale: .4, roughness: .95, sheen: .8, sheenRoughness: .8, sheenColor: 0xffffff });
  const sweater = new THREE.Mesh(wrapGeo(pts, .13), wool);  // oversized
  fly.body.add(sweater);
  const fuzz = new THREE.Mesh(wrapGeo(pts, .17), new THREE.MeshStandardMaterial({ color: 0xf6f3ec, alphaMap: speckle, transparent: true,
    opacity: .28, depthWrite: false, roughness: 1 }));  // mohair halo of loose fibres round the knit (light, or the pattern greys out)
  fuzz.userData.noShadow = true;
  fly.body.add(fuzz);
  n['thorax-black'].visible = n['head-black'].visible = n['antenna_left-black'].visible = n['antenna_right-black'].visible = false;  // bristles under the wool / poking through the lenses
  const cuffTex = fairIsle([[0, 1]]); cuffTex.wrapS = THREE.RepeatWrapping;
  const cuff = new THREE.MeshPhysicalMaterial({ map: cuffTex, bumpMap: speckle, bumpScale: .8, roughness: .95, sheen: 1, sheenRoughness: .7, sheenColor: 0xfff1dc });
  for (const sd of ['left', 'right']) {  // sleeves down the front legs, patterned cuff on the femur
    const fb = local(meshBox([n[`femur_T1_${sd}-body`]])), fy0 = fb.min.y, fy1 = fb.max.y, fm = n[`femur_T1_${sd}-body`];
    const fc = fb.getCenter(V(0, 0, 0));
    fm.add(new THREE.Mesh(shell(fm, .035, W, p => [Math.atan2(p.x - fc.x, p.z - fc.z) / Math.PI + 1, (p.y - fy0) / (fy1 - fy0)]), cuff));
  }

  // --- layered necklaces on the chest: a thin gold chain and three strands of wooden / olive / dark beads ---
  const sp = sweater.geometry.attributes.position, wp = Array.from({ length: sp.count }, (_, i) => V(0, 0, 0).fromBufferAttribute(sp, i));
  const front = (x, y) => wp.reduce((z, p) => Math.abs(p.x - x) < .08 && Math.abs(p.y - y) < .08 ? Math.max(z, p.z) : z, -9);
  const neckY = Math.min(tbox.max.y, hc.y - hs.y * .7), w = (tbox.max.x - tbox.min.x) * .28;
  const necklace = dropY => new THREE.CatmullRomCurve3([-1, -.6, -.3, 0, .3, .6, 1].map(t => {
    const x = t * w, y = neckY - dropY * (1 - t * t) - .02;
    return V(x, y, Math.max(front(x, y), -8) + .07);  // on top of the mohair halo
  }));
  const gold = std(0xe0b050, { metalness: 1, roughness: .25 });
  shiny.push(gold);
  fly.body.add(new THREE.Mesh(new THREE.TubeGeometry(necklace(.3), 48, .007, 5), gold));
  const beads = [], C = c => new THREE.Color(c);
  const strands = [  // [drop, bead count, radius, colors]: a short dark choker-ish one, a wooden one, a long olive/wood one
    [.1, 34, .016, [C('#2e241c'), C('#5a4630')]],
    [.24, 30, .024, [C('#8a5a32'), C('#6e4526'), C('#a0703e')]],
    [.42, 36, .022, [C('#6b7a3a'), C('#8a5a32'), C('#58662e'), C('#e2d6b8')]],
  ];
  for (const [drop, count, r, palette] of strands) {
    const curve = necklace(drop);
    for (let i = 0; i <= count; i++) {
      const bd = new THREE.SphereGeometry(r, 8, 6).translate(...curve.getPoint(i / count).toArray()), c = palette[i % palette.length];
      bd.setAttribute('color', new THREE.Float32BufferAttribute(Array(bd.attributes.position.count).fill(0).flatMap(() => [c.r, c.g, c.b]), 3));
      beads.push(bd);
    }
  }
  fly.body.add(new THREE.Mesh(mergeGeometries(beads), std(0xffffff, { vertexColors: true, roughness: .45 })));
}

// ---------- instruments ----------
const glow =color => std(color, { roughness: .3, metalness: .4, emissive: new THREE.Color(color), emissiveIntensity: 0 });
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
  const shine = () => { for (const m of shiny) { m.envMap = env; m.envMapIntensity = m.metalness ? .6 : .3; } };
  shine();

  // the flybody model replaces the procedural flies once loaded (no /fly.glb = they stay)
  new GLTFLoader().load('/fly.glb', gltf => {
    const proto = gltf.scene;
    flyMaterials(proto);
    for (const m of Object.values(band)) {
      const old = m.fly, fly = makeFlybody(proto, m.color, m.key, old.g.position);
      if (m.key === 'vocals') dressKevinFly(fly);
      fly.g.traverse(o => { if (o.isMesh) o.castShadow = !o.userData.noShadow; });
      m.root.remove(old.g); m.root.add(fly.g); m.fly = fly;
    }
    shine();
    console.log('stage: flybody flies loaded');
  }, undefined, err => console.log('stage: no /fly.glb, procedural flies', err?.message || ''));

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
      fly.animate(m, t, dt, beat);
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
