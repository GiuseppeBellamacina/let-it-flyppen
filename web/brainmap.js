// "Inside the fly brain" panel: the selected fly's simulated neurons, lit by their real spikes every sixteenth.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MEMBERS, ORDER } from './stage.js';

const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = m => typeof m === 'number' ? NOTE[m % 12] + (Math.floor(m / 12) - 1) : String(m);
const unb64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const CLASS = {  // super_class -> color (Cell colors mode)
  sensory: '#ffd23f', central: '#8f6bff', optic: '#3d9bff', descending: '#ff4f9a', ascending: '#ff7a3d',
  motor: '#7dff9b', visual_projection: '#4fe3ff', visual_centrifugal: '#c08bff',
};
const OTHER = '#9d90b3';
const SHORT = { descending: 'desc.', ascending: 'asc.', visual_projection: 'vis. proj.', visual_centrifugal: 'vis. centr.' };
const SPIKES_FULL = 6;  // spikes in a sixteenth that light a neuron fully
const TAU = .22;        // seconds: how fast a flash fades

const VERT = `
attribute float aAct; attribute float aKind; attribute vec3 aCls;
uniform float uScale, uMode;
varying vec3 vCol; varying float vA;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.);
  gl_Position = projectionMatrix * mv;
  float a = aAct, size;
  if (aKind < .5) { vCol = vec3(.32, .42, .75); vA = .11; size = .016; }  // silhouette of the whole brain
  else if (uMode < .5) {                                                 // live spikes: dark blue -> cyan -> white
    vCol = a < .5 ? mix(vec3(.12, .28, .85), vec3(.15, .95, 1.), a * 2.) : mix(vec3(.15, .95, 1.), vec3(1.), a * 2. - 1.);
    vA = .3 + .7 * a; size = .022 + .05 * a;
  } else { vCol = aCls * (.55 + .8 * a) + vec3(.45 * a * a); vA = .5 + .5 * a; size = .024 + .04 * a; }
  gl_PointSize = size * uScale / -mv.z;
}`;
const FRAG = `
varying vec3 vCol; varying float vA;
void main() {
  vec2 p = gl_PointCoord - .5; float d = dot(p, p) * 4.;
  if (d > 1.) discard;
  gl_FragColor = vec4(vCol, vA * exp(-d * 3.));
}`;

export function createBrainMap(el) {
  const $ = s => el.querySelector(s);
  const canvas = $('.bm-view canvas'), wrap = $('.bm-view');
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(30, 2, .1, 20);
  camera.position.set(0, 0, 2.15);
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enableZoom: false, enablePan: false, enableDamping: true, rotateSpeed: .6 });
  const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, uniforms: { uScale: { value: 300 }, uMode: { value: 0 } } });
  const pivot = new THREE.Group(); scene.add(pivot);

  let sel = 'drums', nShape = 0, nAlive = 0, act = null, actAttr = null, classes = [], loading = null, last = 0;

  // toggles: Live spikes / Cell colors
  const legend = $('.bm-legend');
  function setMode(m) {
    mat.uniforms.uMode.value = m;
    for (const b of el.querySelectorAll('.bm-seg button')) { const on = +b.dataset.view === m; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); }
    if (m === 0) legend.innerHTML = '<span class="bm-grad"></span> quiet → firing';
    else {
      const n = {}; for (const c of classes) n[c] = (n[c] ?? 0) + 1;
      legend.innerHTML = Object.entries(n).sort((a, b) => b[1] - a[1])
        .map(([c]) => `<span><i style="background:${CLASS[c] ?? OTHER}"></i>${SHORT[c] ?? c}</span>`).join('');
    }
  }
  for (const b of el.querySelectorAll('.bm-seg button')) b.onclick = () => setMode(+b.dataset.view);
  setMode(0);

  async function load(url) {
    const d = await (await fetch(url)).json();
    nShape = d.n_shape; nAlive = d.n_alive; classes = d.classes;
    const n = nShape + nAlive, pos = new Float32Array(n * 3), kind = new Float32Array(n), cls = new Float32Array(n * 3);
    pos.set(new Float32Array(unb64(d.shape).buffer)); pos.set(new Float32Array(unb64(d.alive).buffer), nShape * 3);
    const box = new THREE.Box3().setFromArray(pos.subarray(0, nShape * 3)), mid = box.getCenter(new THREE.Vector3());
    for (let i = 0; i < n; i++) { pos[i * 3] -= mid.x; pos[i * 3 + 1] = mid.y - pos[i * 3 + 1]; pos[i * 3 + 2] -= mid.z; }  // y points down in FlyWire
    const c = new THREE.Color();
    for (let i = 0; i < nAlive; i++) { kind[nShape + i] = 1; c.set(CLASS[classes[i]] ?? OTHER).toArray(cls, (nShape + i) * 3); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aKind', new THREE.BufferAttribute(kind, 1));
    geo.setAttribute('aCls', new THREE.BufferAttribute(cls, 3));
    actAttr = new THREE.BufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aAct', actAttr);
    act = actAttr.array.subarray(nShape);
    pivot.add(new THREE.Points(geo, mat));
    $('.bm-n').textContent = `${nAlive.toLocaleString('en')} of 139,255 neurons · FlyWire v783`;
    setMode(mat.uniforms.uMode.value);
  }

  // counters and neural readout
  const nSpk = $('.bm-spikes'), nAct = $('.bm-active'), bars = $('.bm-bars');
  function readout(msg) {
    const top = msg.scores?.[sel] ?? [], played = new Set((msg.play?.[sel] ?? []).map(n => n[0]));
    bars.innerHTML = top.length ? top.map(([c, s]) => `<li class="${played.has(c) ? 'hit' : ''}"><span>${noteName(c)}</span>
      <i><b style="width:${Math.max(0, Math.min(1, s)) * 100}%"></b></i><em>${s.toFixed(2)}</em></li>`).join('')
      : '<li class="none">no readout yet</li>';
  }

  // render loop: decay the flashes, sway, draw; paused when the panel is not visible
  function frame(t) {
    requestAnimationFrame(frame);
    const dt = Math.min(.1, (t - last) / 1000); last = t;
    if (document.hidden || !canvas.offsetParent || el.closest('.hidden') || !act) return;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (canvas.width !== Math.round(W * renderer.getPixelRatio()) || canvas.height !== Math.round(H * renderer.getPixelRatio())) {
      renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix();
      mat.uniforms.uScale.value = H * renderer.getPixelRatio() / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    }
    const k = Math.exp(-dt / TAU);
    for (let i = 0; i < nAlive; i++) act[i] *= k;
    actAttr.needsUpdate = true;
    pivot.rotation.y = .55 * Math.sin(t / 7000);  // slow sway, so left/right labels stay true
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);

  return {
    onHello(h) { if (h.brain && !loading) loading = load(h.brain).catch(e => { loading = null; console.warn('brain map:', e); }); },
    select(k) {
      sel = k;
      el.style.setProperty('--c', MEMBERS[k].color);
      if (act) act.fill(0);
    },
    onStep(msg) {
      readout(msg);
      if (!act || !msg.spikes) return;
      const all = unb64(msg.spikes), off = ORDER.indexOf(sel) * nAlive;
      let sum = 0, on = 0;
      for (let i = 0; i < nAlive; i++) {
        const c = all[off + i];
        if (c) { sum += c; on++; act[i] = Math.max(act[i], Math.min(1, c / SPIKES_FULL)); }
      }
      nSpk.textContent = sum.toLocaleString('en');
      nAct.textContent = on.toLocaleString('en');
    },
  };
}
