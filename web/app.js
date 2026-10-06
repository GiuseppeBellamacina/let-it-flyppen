// Server connection, audio (Tone.js) and UI. The 3D stage lives in stage.js.
import { createStage, MEMBERS, ORDER } from './stage.js';

const $ = s => document.querySelector(s);
const pct = x => (x == null || Number.isNaN(x)) ? '–' : `${Math.round(x * 100)}%`;
const f1of = (tp, fp, fn) => 2 * tp / Math.max(1, 2 * tp + fp + fn);
const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = m => typeof m === 'number' ? NOTE[m % 12] + (Math.floor(m / 12) - 1) : String(m);
const hash = new URLSearchParams(location.hash.slice(1));

const stage = createStage(document.body);
const band = Object.fromEntries(ORDER.map(k => [k, {
  key: k, ...MEMBERS[k], tp: 0, fp: 0, fn: 0, nPlay: 0, nExp: 0, history: [], prec: null, rec: null,
  act: 0, actMax: 1, muted: false, solo: false, roll: [], ras: [], rasMax: 4, lut: null,
}]));

// ---------- audio ----------
const master = new Tone.Volume(-4).toDestination();
const verb = new Tone.Reverb({ decay: 3.2, wet: .28 }).connect(master);
const phaser = new Tone.Phaser({ frequency: .25, octaves: 3, baseFrequency: 500, wet: .5 }).connect(verb);
const chorus = new Tone.Chorus(2, 3, .4).connect(verb).start();
const SYN = {
  kick: new Tone.MembraneSynth({ pitchDecay: .03, octaves: 6, envelope: { attack: .001, decay: .35, sustain: 0 }, volume: -4 }).connect(master),
  snare: new Tone.NoiseSynth({ noise: { type: 'white' }, envelope: { attack: .001, decay: .15, sustain: 0 }, volume: -15 }).connect(verb),
  hat: new Tone.MetalSynth({ envelope: { attack: .001, decay: .05, release: .01 }, harmonicity: 5.1, modulationIndex: 32,
    resonance: 4000, octaves: 1.5, volume: -28 }).connect(master),
  bass: new Tone.MonoSynth({ oscillator: { type: 'sawtooth' }, filter: { Q: 2, type: 'lowpass' },
    filterEnvelope: { baseFrequency: 90, octaves: 3, attack: .01, decay: .2, sustain: .3 },
    envelope: { attack: .005, decay: .2, sustain: .6, release: .2 }, volume: -8 }).connect(master),
  guitar: new Tone.PolySynth(Tone.Synth, { oscillator: { type: 'fattriangle', count: 2, spread: 18 },
    envelope: { attack: .004, decay: .4, sustain: .2, release: .6 }, volume: -14 }).connect(chorus),
  synth: new Tone.PolySynth(Tone.Synth, { oscillator: { type: 'fatsawtooth', count: 3, spread: 30 },
    envelope: { attack: .02, decay: .3, sustain: .4, release: .8 }, volume: -20 }).connect(phaser),
  vocals: new Tone.Synth({ oscillator: { type: 'fmsine', modulationIndex: 2 },
    envelope: { attack: .04, decay: .2, sustain: .7, release: .5 }, volume: -10 }).connect(verb),
};
let listen = 'play', stepSec = .12, audioOn = false;
const freq = m => Tone.Frequency(m, 'midi').toFrequency();
function sound(key, notes, t) {
  for (const [c, dur] of notes) {
    const d = Math.max(.05, dur * stepSec * .95);
    if (key === 'drums') {
      if (c === 'kick') SYN.kick.triggerAttackRelease('C1', '8n', t);
      if (c === 'snare') SYN.snare.triggerAttackRelease('16n', t);
      if (c === 'hat') SYN.hat.triggerAttackRelease(300, '32n', t, .6);
    } else SYN[key].triggerAttackRelease(freq(c), d, t);
  }
}
const audible = m => ORDER.some(k => band[k].solo) ? m.solo : !m.muted;
// "Original": the real recording, kept in time with the sixteenth the band is playing (msg.time)
const rec = new Audio();
rec.preload = 'auto';
function followRecording(t) {
  if (listen !== 'record' || !audioOn || t == null) return rec.pause();
  if (rec.paused || Math.abs(rec.currentTime - t) > .15) rec.currentTime = t;  // jump at rehearsal start, or drift
  if (rec.paused) rec.play().catch(() => {});
}

// ---------- member cards ----------
let sel = 'drums';
for (const [n, k] of ORDER.entries()) {
  const m = band[k], card = document.createElement('article');
  card.className = 'card panel'; card.style.setProperty('--c', m.color);
  card.innerHTML = `
    <div class="card-head">
      <button class="pick" title="Focus on ${m.name} (${n + 1})"><span class="name">${m.name}</span>
        <span class="live"><small>F1</small><span class="v">–</span></span></button>
      <button class="tog m" aria-pressed="false" title="Mute">M</button>
      <button class="tog s" aria-pressed="false" title="Solo">S</button>
    </div>
    <div class="stats">
      <span>last <b class="last">–</b></span><span>best <b class="best">–</b></span>
      <span>P <b class="p">–</b> · R <b class="r">–</b></span><span>notes <b class="notes">0/0</b></span>
    </div>
    <div class="card-foot">
      <div class="meter" title="brain activity"><i></i></div>
      <svg class="spark" viewBox="0 0 92 22" preserveAspectRatio="none" aria-hidden="true">
        <line x1="0" x2="92" y1="11" y2="11" stroke="rgba(255,255,255,.08)"/>
        <polyline fill="none" stroke="${m.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>
    </div>`;
  const q = s => card.querySelector(s);
  m.el = { card, live: q('.v'), last: q('.last'), best: q('.best'), p: q('.p'), r: q('.r'), notes: q('.notes'),
    meter: q('.meter i'), spark: q('polyline'), mute: q('.m'), solo: q('.s') };
  card.onclick = () => select(k, true);
  q('.m').onclick = e => { e.stopPropagation(); m.muted = !m.muted; refreshMix(); };
  q('.s').onclick = e => { e.stopPropagation(); m.solo = !m.solo; refreshMix(); };
  $('#cards').append(card);
}
function refreshMix() {
  for (const m of Object.values(band)) {
    m.el.mute.setAttribute('aria-pressed', m.muted); m.el.solo.setAttribute('aria-pressed', m.solo);
    m.el.card.classList.toggle('silent', !audible(m));
  }
}
function select(k, cam, instant = false) {
  sel = k;
  for (const m of Object.values(band)) m.el.card.classList.toggle('sel', m.key === k);
  $('#roll-name').textContent = $('#raster-name').textContent = band[k].name;
  if (cam) stage.focus(k, instant);
  drawRaster();
}
function showCard(m) {
  const n = m.tp + m.fp + m.fn, last = m.history.at(-1);
  m.el.live.textContent = n ? pct(f1of(m.tp, m.fp, m.fn)) : '–';
  m.el.last.textContent = pct(last);
  m.el.best.textContent = m.history.length ? pct(Math.max(...m.history)) : '–';
  m.el.p.textContent = n ? pct(m.tp / Math.max(1, m.tp + m.fp)) : pct(m.prec);
  m.el.r.textContent = n ? pct(m.tp / Math.max(1, m.tp + m.fn)) : pct(m.rec);
  m.el.notes.textContent = `${m.nPlay}/${m.nExp}`;
}
function drawSpark(m) {
  const h = m.history.slice(-60), n = Math.max(h.length - 1, 1);
  m.el.spark.setAttribute('points', h.map((v, i) => `${(i / n) * 92},${21 - (v ?? 0) * 20}`).join(' '));
}

// ---------- average F1 chart ----------
function drawChart() {
  const svg = $('#chart'), W = 320, H = 130, L = 30, R = 10, T = 8, B = 20, n = band.drums.history.length;
  const x = i => L + (n > 1 ? i / (n - 1) : .5) * (W - L - R), y = v => T + (1 - v) * (H - T - B);
  const stride = Math.ceil(n / 300);  // with hundreds of quick rehearsals: one point every `stride`
  const line = arr => arr.map((v, i) => [i, v]).filter(([i]) => i % stride === 0 || i === n - 1)
    .map(([i, v]) => `${x(i).toFixed(1)},${y(v ?? 0).toFixed(1)}`).join(' ');
  const avg = band.drums.history.map((_, i) => ORDER.reduce((s, k) => s + (band[k].history[i] ?? 0), 0) / ORDER.length);
  let out = '';
  for (const v of [0, .25, .5, .75, 1]) out += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="rgba(255,255,255,${v % .5 ? .05 : .12})"/>
    ${v % .5 ? '' : `<text x="${L - 5}" y="${y(v) + 3}" text-anchor="end">${v * 100}%</text>`}`;
  if (!n) out += `<text x="${(L + W - R) / 2}" y="${H / 2}" text-anchor="middle">no rehearsal completed yet</text>`;
  else {
    const ticks = [...new Set([0, Math.floor((n - 1) / 2), n - 1])];
    for (const i of ticks) out += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">#${i + 1}</text>`;
    for (const k of ORDER) out += `<polyline fill="none" stroke="${band[k].color}" stroke-opacity=".38" stroke-width="1" points="${line(band[k].history)}"/>`;
    const pts = line(avg);
    out += `<polyline fill="url(#gfill)" stroke="none" points="${x(0)},${y(0)} ${pts} ${x(n - 1)},${y(0)}"/>
      <polyline fill="none" stroke="#f4ead9" stroke-width="2.2" stroke-linejoin="round" points="${pts}"/>
      <circle cx="${x(n - 1)}" cy="${y(avg[n - 1])}" r="3" fill="#f4ead9"/>
      <text x="${Math.min(x(n - 1), W - R - 20)}" y="${y(avg[n - 1]) < 22 ? y(avg[n - 1]) + 15 : y(avg[n - 1]) - 7}" text-anchor="middle" style="fill:#f4ead9;font-weight:600">${pct(avg[n - 1])}</text>`;
  }
  svg.innerHTML = `<defs><linearGradient id="gfill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ff4f9a" stop-opacity=".35"/>
    <stop offset="1" stop-color="#8f6bff" stop-opacity="0"/></linearGradient></defs>${out}
`;
}

// ---------- piano roll: score (ghost) vs played (green correct, red wrong) ----------
const roll = $('#roll'), rctx = roll.getContext('2d');
const VIS = 64;  // visible sixteenths = 4 bars
let tick = 0, lastStepAt = 0, range = {};
function drawRoll() {
  const dpr = Math.min(devicePixelRatio, 2), W = roll.clientWidth, H = roll.clientHeight;
  if (!W) return;
  if (roll.width !== Math.round(W * dpr)) { roll.width = Math.round(W * dpr); roll.height = Math.round(H * dpr); }
  rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  rctx.clearRect(0, 0, W, H);
  const m = band[sel], cls = meta?.parts?.[sel]?.classes ?? [], drums = sel === 'drums';
  const G = 34, colW = (W - G - 6) / VIS;
  const now = tick + Math.min(1, (performance.now() - lastStepAt) / (stepSec * 1000));
  const X = t => W - 6 - (now - t) * colW;
  // rows: the 3 voices for drums; for the others, the range of recently seen notes
  let rows;
  if (drums) rows = ['hat', 'snare', 'kick'];
  else {
    let lo = Infinity, hi = -Infinity;
    for (const e of m.roll) for (const [c] of [...e.play, ...e.target]) if (typeof c === 'number') { lo = Math.min(lo, c); hi = Math.max(hi, c); }
    if (lo === Infinity) { const mid = cls.length ? cls[cls.length >> 1] : 60; lo = mid - 6; hi = mid + 6; }
    if (hi - lo < 11) { const pad = (11 - (hi - lo)) / 2; lo = Math.floor(lo - pad); hi = Math.ceil(hi + pad); }
    const r = range[sel] ??= { lo, hi };
    r.lo += (lo - r.lo) * .08; r.hi += (hi - r.hi) * .08;
    const a = Math.floor(r.lo), b = Math.ceil(r.hi);
    rows = Array.from({ length: b - a + 1 }, (_, i) => b - i);
  }
  const rh = Math.min(26, (H - 14) / rows.length), top = (H - 14 - rh * rows.length) / 2, rowOf = c => rows.indexOf(c);
  rctx.save(); rctx.translate(0, top);
  // background: darker black keys, beat and bar grid
  rows.forEach((c, i) => {
    if (typeof c === 'number' && [1, 3, 6, 8, 10].includes(c % 12)) { rctx.fillStyle = 'rgba(0,0,0,.28)'; rctx.fillRect(G, i * rh, W - G, rh); }
    if (drums || rh >= 9 || c % 12 === 0) {
      rctx.fillStyle = '#9d90b3'; rctx.font = `${Math.min(10, Math.max(8, rh * .8))}px JetBrains Mono, monospace`;
      rctx.textBaseline = 'middle'; rctx.fillText(noteName(c), 3, i * rh + rh / 2);
    }
  });
  rctx.beginPath(); rctx.rect(G, -top, W - G, H); rctx.clip();
  for (const e of m.roll) if (e.i % 4 === 0) {
    const x = X(e.tick), bar = e.i % 16 === 0;
    rctx.fillStyle = bar ? 'rgba(255,255,255,.16)' : 'rgba(255,255,255,.05)'; rctx.fillRect(x, -top, 1, H - 14);
    if (bar) { rctx.fillStyle = '#9d90b3'; rctx.font = '9px JetBrains Mono, monospace'; rctx.fillText(Math.floor(e.i / 16) + 1, x + 3, H - 6 - top); }
  }
  for (const e of m.roll) {
    const x = X(e.tick); if (x < G - 200) continue;
    const want = new Set(e.target.map(n => n[0]));
    for (const [c, d] of e.target) {
      const r = rowOf(c); if (r < 0) continue;
      rctx.fillStyle = 'rgba(255,255,255,.07)'; rctx.strokeStyle = 'rgba(255,255,255,.5)'; rctx.lineWidth = 1;
      rctx.fillRect(x, r * rh + 1, d * colW - 1, rh - 2); rctx.strokeRect(x + .5, r * rh + 1.5, d * colW - 2, rh - 3);
    }
    for (const [c, d] of e.play) {
      const r = rowOf(c); if (r < 0) continue;
      const ok = want.has(c), inset = Math.min(2, rh * .2);
      rctx.fillStyle = ok ? '#7dff9b' : '#ff5470'; rctx.shadowColor = rctx.fillStyle; rctx.shadowBlur = 6;
      rctx.fillRect(x + inset, r * rh + 1 + inset, Math.max(2, d * colW - 1 - 2 * inset), rh - 2 - 2 * inset);
      rctx.shadowBlur = 0;
    }
  }
  rctx.restore();
  const g = rctx.createLinearGradient(W - 40, 0, W, 0);  // "now" on the right edge
  g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, m.color + '55');
  rctx.fillStyle = g; rctx.fillRect(W - 40, 0, 40, H - 14);
  if (!m.roll.length) { rctx.fillStyle = '#9d90b3'; rctx.font = '11px JetBrains Mono, monospace'; rctx.textAlign = 'center';
    rctx.fillText(mode === 'fast' ? 'quick rehearsals: no live notes' : 'waiting for live rehearsals…', (W + G) / 2, H / 2); rctx.textAlign = 'left'; }
}

// ---------- raster: output neuron spikes, scrolling heat map ----------
const rasterCanvas = $('#raster'), xctx = rasterCanvas.getContext('2d'), img = xctx.createImageData(128, 48);
let rasterSeen = false;
function lut(hex) {  // background → member color → white
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)), bg = [11, 7, 20], out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255, a = Math.min(1, v / .6), b = Math.max(0, (v - .6) / .4);
    for (let j = 0; j < 3; j++) out[i * 3 + j] = bg[j] + (c[j] - bg[j]) * a + (255 - c[j]) * b;
  }
  return out;
}
function drawRaster() {
  const m = band[sel], L = m.lut ??= lut(m.color), d = img.data, off = 128 - m.ras.length;
  for (let x = 0; x < 128; x++) {
    const col = m.ras[x - off];
    for (let y = 0; y < 48; y++) {
      const v = col ? Math.min(255, Math.round(255 * Math.sqrt((col[y] ?? 0) / m.rasMax))) : 0, p = (y * 128 + x) * 4;
      d[p] = L[v * 3]; d[p + 1] = L[v * 3 + 1]; d[p + 2] = L[v * 3 + 2]; d[p + 3] = 255;
    }
  }
  xctx.putImageData(img, 0, 0);
}

// ---------- header and info ----------
let meta = null, pass = 0, mode = 'wait', started = false;
function setMode(m) {
  mode = m;
  $('#mode').textContent = { live: 'live', fast: 'quick rehearsals', wait: 'waiting', off: 'server offline' }[m];
  $('#mode').className = m;
}
function showInfo(h) {
  const n = v => v == null ? '?' : Number(v).toLocaleString('en');
  $('#info-body').innerHTML = `
    <p>Five flies, five separate brains. Each one is the <b>FlyWire v783</b> connectome of <i>Drosophila</i>, simulated with the
    leaky integrate-and-fire (LIF) model of Shiu et al. 2024: <b>${n(h.neurons)}</b> active neurons per fly.</p>
    <p>The flies "hear" the song through <b>${n(h.ears)}</b> antenna neurons (Johnston's organ), stimulated at every sixteenth note.
    From what happens inside, <b>${n(h.outputs)}</b> output neurons are read${h.features ? `, yielding <b>${n(h.features)}</b> features` : ''}.</p>
    <p>Honestly: the brain is <b>frozen</b>. The connectome's synapses never change. Only the <b>readout</b> learns, a linear
    mapping from outputs to notes${h.learner ? ` (updated with <b>${h.learner}</b>)` : ''}. The fly acts as a reservoir of dynamics;
    the final layer picks the "music".</p>
    <p><b>F1</b> = harmonic mean of precision (P: how many played notes were correct) and recall (R: how many expected notes were played),
    on the exact sixteenth, ignoring duration. The song is <i>Let It Happen</i> by Tame Impala, <b>${h.bpm}</b> bpm, one sixteenth every
    ${Math.round(h.step_sec * 1000)} ms; the rehearsed section is ${section(h)}.</p>
    <p>"Quick rehearsals" = fast passes over the same recording already heard (the brain is deterministic, simulating it once is enough);
    "live" = the real simulation, in time with the music, the one you hear.</p>`;
}
const section = h => `bars ${h.steps[0] / 16 + 1}–${h.steps[1] / 16}`;
function toggleInfo(open = $('#info').hidden) {
  $('#info').hidden = !open; $('#info-btn').setAttribute('aria-expanded', open);
  if (open) $('#info .close').focus();
}
$('#info-btn').onclick = () => toggleInfo();
$('#info .close').onclick = () => { toggleInfo(false); $('#info-btn').focus(); };

// ---------- server messages ----------
function onHello(h) {
  meta = h; stepSec = h.step_sec; pass = h.history?.length ?? 0;
  const recBtn = $('.seg [data-mode="record"]');
  recBtn.disabled = !h.audio;
  if (h.audio) { if (!rec.src.endsWith(h.audio)) rec.src = h.audio; }
  else { recBtn.title = 'No recording: the song comes from a MIDI file'; if (listen === 'record') setListen('play'); }
  for (const k of ORDER) {
    const m = band[k];
    m.history = (h.history ?? []).map(r => r[k] ?? 0);
    drawSpark(m); showCard(m);
  }
  $('#pass').textContent = pass ? `#${pass}` : '–';
  $('#section').textContent = section(h);
  $('#bpm').textContent = `${h.bpm} bpm`;
  showInfo(h); drawChart();
}
function onPass(p) {
  pass = p.n; $('#pass').textContent = `#${p.n}${p.fast ? '' : ' ✓'}`;
  if (p.fast) { setMode('fast'); $('#progress').style.width = '100%'; }
  for (const k of ORDER) {
    const m = band[k];
    m.history.push(p.f1?.[k] ?? 0);
    m.prec = p.precision?.[k] ?? null; m.rec = p.recall?.[k] ?? null;
    if (p.fast) Object.assign(m, { tp: 0, fp: 0, fn: 0 });
    drawSpark(m); showCard(m);
  }
  drawChart();
}
function onStart(s) {
  pass = s.n; started = true; setMode('live');
  $('#pass').textContent = `#${s.n}`;
  for (const k of ORDER) { Object.assign(band[k], { tp: 0, fp: 0, fn: 0, nPlay: 0, nExp: 0 }); showCard(band[k]); }
}
const nk = n => JSON.stringify(n[0]);
function onStep(msg) {
  if (mode !== 'live') { setMode('live'); if (!started) { started = true; $('#pass').textContent = `#${pass + 1}`; } }  // connected mid-rehearsal: it is the one after those already finished
  tick++; lastStepAt = performance.now();
  const t = audioOn ? Tone.now() + .06 : 0;
  if (meta) {
    const [s0, s1] = meta.steps, f = (msg.i - s0 + 1) / Math.max(1, s1 - s0);
    $('#progress').style.width = `${Math.min(100, f * 100)}%`;
    $('#progress').parentElement.setAttribute('aria-valuenow', Math.round(f * 100));
  }
  $('#pos').textContent = `bar ${Math.floor(msg.i / 16) + 1} · ${Math.floor(msg.i % 16 / 4) + 1}/4`;
  if (msg.i % 4 === 0) stage.onBeat();
  stage.hearStep();
  let f1sum = 0;
  const levels = [], played = [];
  ORDER.forEach((k, i) => {
    const m = band[k], play = msg.play?.[k] ?? [], target = msg.target?.[k] ?? [];
    const tset = new Set(target.map(nk)), hit = play.filter(n => tset.has(nk(n))).length;
    m.tp += hit; m.fp += play.length - hit; m.fn += target.length - hit;
    m.nPlay += play.length; m.nExp += target.length;
    m.act = msg.act?.[i] ?? 0; m.actMax = Math.max(m.actMax * .995, m.act, 1e-6);
    const heard = listen === 'play' ? play : target;
    if (audioOn && audible(m) && listen !== 'record') sound(k, heard, t);
    stage.perform(k, heard);
    stage.setActivity(k, m.act / m.actMax);
    levels.push(m.act / m.actMax); played.push(play.length > 0);
    m.roll.push({ tick, i: msg.i, play, target }); if (m.roll.length > VIS + 8) m.roll.shift();
    const ras = msg.raster?.[k];
    if (ras) {
      rasterSeen = true; m.ras.push(ras); if (m.ras.length > 128) m.ras.shift();
      m.rasMax = Math.max(m.rasMax * .99, ...ras, 1);
    }
    showCard(m); m.el.meter.style.width = `${100 * m.act / m.actMax}%`;
    f1sum += f1of(m.tp, m.fp, m.fn);
  });
  followRecording(msg.time);
  $('#raster-empty').hidden = rasterSeen;
  if (!rasterSeen) $('#raster-empty').textContent = 'the server is not sending the raster';
  stage.setEnergy(.15 + .85 * f1sum / ORDER.length);
  stage.monitorColumn(levels, played);
  drawRaster();
}
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('#status').textContent = 'connected'; };
  ws.onclose = () => { setMode('off'); $('#status').textContent = 'server offline, retrying…'; setTimeout(connect, 2000); };
  ws.onmessage = e => {
    let msg; try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.t === 'hello') onHello(msg);
    else if (msg.t === 'start') onStart(msg);
    else if (msg.t === 'pass') onPass(msg);
    else if (msg.t === 'status') $('#status').textContent = msg.text || (mode === 'live' ? 'the flies are playing' : '');
    else if (msg.t === 'step') onStep(msg);
  };
}

// ---------- controls ----------
function setListen(v) {
  listen = v;
  if (v !== 'record') rec.pause();
  for (const b of document.querySelectorAll('.seg button')) { b.classList.toggle('on', b.dataset.mode === v); b.setAttribute('aria-pressed', b.dataset.mode === v); }
}
for (const b of document.querySelectorAll('.seg button')) b.onclick = () => setListen(b.dataset.mode);
$('#home').onclick = () => stage.focus(null);
const ui = $('#ui');
function measure() { if (!ui.classList.contains('hidden')) { const r = $('#free').getBoundingClientRect(); stage.setFrame(r.width > 40 ? r : null); } }
function toggleUI() {
  const hidden = ui.classList.toggle('hidden');
  $('#shown-hint').hidden = !hidden; stage.setUIHidden(hidden);
  if (!hidden) measure();
}
new ResizeObserver(measure).observe($('#free'));
addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, textarea, select')) return;
  const k = e.key.toLowerCase();
  if (k >= '1' && k <= '5') select(ORDER[+k - 1], true);
  else if (k === 'o') {  // Flies -> Score -> Original (if there is a recording) -> Flies
    const modes = ['play', 'target', ...(meta?.audio ? ['record'] : [])];
    setListen(modes[(modes.indexOf(listen) + 1) % modes.length]);
  }
  else if (k === 'h') toggleUI();
  else if (k === 'i') toggleInfo();
  else if (k === 'escape') { if (!$('#info').hidden) toggleInfo(false); else stage.focus(null); }
  else return;
  e.preventDefault();
});
$('#enter').onclick = async () => { await Tone.start(); audioOn = true; $('#gate').remove(); };
if (hash.has('demo')) $('#gate').remove();  // visuals only, no audio (displays, screenshots)

(function loop() { drawRoll(); requestAnimationFrame(loop); })();
measure();
select(hash.get('focus') in band ? hash.get('focus') : 'drums', hash.get('focus') in band, true);
drawChart();
if (hash.has('info')) toggleInfo(true);
connect();
