"""Prepare the data for the fly band.

  uv run prep.py fetch                 # FlyWire v783 connectome + annotations
  uv run prep.py song song.mp3         # mp3 -> stems (demucs) -> notes (basic-pitch) -> data/song.json
  uv run prep.py song song.mid         # or from a multitrack MIDI (cleaner targets)
  uv run prep.py check                 # check that the stems in data/stems are complete and consistent with the mix
  uv run prep.py score                 # how close each transcribed part is to its stem (chroma similarity)
  uv run prep.py samples               # re-cut the instrument samples from the stems (song does it already)

"song" and "check" need the prep extra: uv sync --extra prep
"""
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
URLS = {
    "Completeness_783.csv": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "Connectivity_783.parquet": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}
PARTS = ["drums", "bass", "guitar", "synth", "vocals"]
MONO = {"bass": min, "guitar": max, "vocals": max}  # monophonic part -> which note to keep if several start together
DRUM_BANDS = {"kick": (20, 150), "snare": (2000, 5000), "hat": (5000, 16000)}  # Hz; snare on its "snares" (2-5k), below that the kick click bleeds in
DRUM_GATE = 0.3  # fraction of the new-energy peak (99th percentile) a hit must exceed in its band
MIN_NOTES = 50  # below this threshold a stem is considered empty/bleed
STEMS = {"drums": "htdemucs_ft", "bass": "htdemucs_ft", "vocals": "htdemucs_ft",  # ft: better separation
         "guitar": "htdemucs_6s", "piano": "htdemucs_6s", "other": "htdemucs_6s"}  # 6s: the only one with guitar/piano
# basic-pitch per stem: instrument range (Hz) and higher thresholds to cut bleed from the other stems
PITCH = {"bass": dict(minimum_frequency=30, maximum_frequency=400, onset_threshold=0.6),
         "vocals": dict(minimum_frequency=80, maximum_frequency=1100, onset_threshold=0.6, frame_threshold=0.4),
         "guitar": dict(minimum_frequency=70, maximum_frequency=1500, onset_threshold=0.5, frame_threshold=0.3),
         "piano": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.5, frame_threshold=0.3),
         "other": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.5, frame_threshold=0.3)}
# (thresholds picked by `score`: synth 0.744 -> 0.801 with these + MAX_POLY 4; vocals 0.496 -> 0.534 with pYIN)
PYIN_RANGE = {"bass": (30, 400), "vocals": (80, 1100)}  # mono parts that also try pYIN; the better one wins
MIN_CLASS_NOTES = 3  # notes appearing fewer times than this are almost always transcription errors
MAX_POLY = 4  # max simultaneous notes for the synth (keeps the loudest)
SAMPLE_LEN = (0.3, 1.5)  # seconds: shortest/longest sample cut from a stem
DRUM_SAMPLE_LEN = {"kick": 0.45, "snare": 0.35, "hat": 0.12}
MONO_MIN_FRAMES = 6  # pYIN notes shorter than this (~70 ms) are glitches


def fetch():
    DATA.mkdir(exist_ok=True)
    for name, url in URLS.items():
        if not (DATA / name).exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, DATA / name)
    print("data ok")


def step_grid(beats, n_after=64):
    """Sixteenth times: each interval between beats split in 4 (follows the song's tempo drift)."""
    import numpy as np
    beats = np.asarray(beats, float)
    beat = float(np.median(np.diff(beats)))
    head = beats[0] - beat * np.arange(int(beats[0] // beat), 0, -1)
    tail = beats[-1] + beat * np.arange(1, n_after)
    beats = np.concatenate([head, beats, tail])
    t = (beats[:-1, None] + np.diff(beats)[:, None] * np.arange(4) / 4).ravel()
    return t, beat / 4


def to_events(notes, grid, step_sec, mono=None):
    """notes: [(start, end, pitch, amplitude, ...)] -> classes (sorted pitches) and
    events [step, class, duration_in_steps, velocity 0-1]."""
    from collections import Counter
    import numpy as np
    common = {p for p, n in Counter(int(n[2]) for n in notes).items() if n >= MIN_CLASS_NOTES}
    loud = np.percentile([n[3] for n in notes], 95) if notes else 1.0
    by_step, amp = {}, {}
    for start, end, pitch, a, *_ in notes:
        if int(pitch) in common:
            s = int(np.argmin(np.abs(grid - start)))
            by_step.setdefault(s, {})[int(pitch)] = max(1, round((end - start) / step_sec))
            amp[s, int(pitch)] = a
    by_step = {s: dict(sorted(ps.items(), key=lambda kv: -amp[s, kv[0]])[:MAX_POLY]) for s, ps in by_step.items()}
    if mono:  # a single note per step, truncated at the next onset
        by_step = {s: {mono(ps): ps[mono(ps)]} for s, ps in by_step.items()}
        steps = sorted(by_step)
        for a, b in zip(steps, steps[1:]):
            p = next(iter(by_step[a]))
            by_step[a][p] = min(by_step[a][p], b - a)
    classes = sorted({p for ps in by_step.values() for p in ps})
    idx = {p: i for i, p in enumerate(classes)}
    events = [[s, idx[p], d, round(float(min(1.0, amp[s, p] / loud)), 2)] for s in sorted(by_step) for p, d in by_step[s].items()]
    return classes, events


def drum_events(y, sr, grid):
    import librosa
    import numpy as np
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    events = []
    for c, (lo, hi) in enumerate(DRUM_BANDS.values()):
        band = S[(freqs >= lo) & (freqs < hi)]
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(band), sr=sr)
        flux = np.maximum(0, np.diff(band.sum(0), prepend=0))  # absolute new energy in the band
        peak = np.percentile(flux, 99)
        for f in librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=False):
            hit = flux[max(0, f - 2):f + 3].max()
            if hit >= DRUM_GATE * peak:  # onset_detect normalizes: drop "hits" that are only bleed from other bands
                events.append((int(np.argmin(np.abs(grid - librosa.frames_to_time(f, sr=sr)))), c, 1,
                               round(float(min(1.0, hit / peak)), 2)))
    best = {}  # one hit per (step, class): the loudest
    for e in events:
        if e[:2] not in best or e[3] > best[e[:2]][3]:
            best[e[:2]] = e
    return sorted(best.values())


def separate(path):
    """mp3 -> stem wavs in data/stems (cached). IO via librosa/soundfile: no ffmpeg/torchaudio."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    out = DATA / "stems"
    if all((out / f"{n}.wav").exists() for n in [*STEMS, "mix"]):
        return out
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = None
    for name in dict.fromkeys(STEMS.values()):
        model = get_model(name).eval()
        if y is None:
            y, sr = librosa.load(path, sr=model.samplerate, mono=False)
            y = np.stack([y, y]) if y.ndim == 1 else y
            sf.write(out / "mix.wav", y.T, sr)
        x = torch.tensor(y)[None]
        ref = x.mean(1)
        print(f"demucs {name} on {dev} (a few minutes)...")
        with torch.no_grad():  # shifts=2: average over 2 time shifts, cleaner separation
            stems = apply_model(model, (x - ref.mean()) / ref.std(), device=dev, shifts=2, split=True,
                                overlap=0.25, progress=True)[0]
        stems = stems * ref.std() + ref.mean()
        for stem, s in zip(model.sources, stems.numpy()):
            if STEMS.get(stem) == name:
                sf.write(out / f"{stem}.wav", s.T, sr)
    check()
    return out


def check():
    """Stems complete (same length as the mix), not silent, and how well their sum rebuilds the mix."""
    import numpy as np
    import soundfile as sf
    out = DATA / "stems"
    mix, sr = sf.read(out / "mix.wav")
    total = np.zeros_like(mix)
    print(f"{'stem':7s} {'length':>7s} {'rms':>8s} {'silent':>9s}   (mix {len(mix) / sr:.1f}s)")
    for name in STEMS:
        y, s = sf.read(out / f"{name}.wav")
        assert s == sr and y.shape == mix.shape, f"{name}: length/format differs from the mix"
        total += y
        m = y.mean(1)[: len(y) // sr * sr].reshape(-1, sr)
        silent = int((20 * np.log10(np.sqrt((m ** 2).mean(1)) + 1e-9) < -50).sum())
        print(f"{name:7s} {len(y) / s:6.1f}s {20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9):6.1f}dB "
              f"{silent:4d}/{len(m)}s")
    res = 20 * np.log10(np.sqrt(((mix - total) ** 2).mean()) / np.sqrt((mix ** 2).mean()))
    print(f"residual mix - stem sum: {res:.1f} dB (rough: the stems come from two models, they don't sum exactly to the mix)")


def mono_notes(stem, fmin, fmax):
    """Monophonic line via pYIN f0: a note per run of the same semitone, split at onsets. -> [(start, end, pitch, amp)]."""
    import librosa
    import numpy as np
    import scipy.ndimage
    y, sr = librosa.load(DATA / "stems" / f"{stem}.wav", sr=22050)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    midi = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1)
    midi = scipy.ndimage.median_filter(midi, 5)  # drop single-frame octave/semitone flickers
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][: len(midi)]
    onsets = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True))
    t = librosa.frames_to_time(np.arange(len(midi) + 1), sr=sr, hop_length=hop)
    notes, start = [], 0
    for f in range(1, len(midi) + 1):
        if f == len(midi) or midi[f] != midi[start] or f in onsets:
            if midi[start] > 0 and f - start >= MONO_MIN_FRAMES:
                notes.append((float(t[start]), float(t[f]), int(midi[start]), float(rms[start:f].mean())))
            start = f
    return notes


PART_STEM = {"bass": "bass", "guitar": "guitar", "synth": "other", "vocals": "vocals"}  # stem each pitched part is scored against


def stem_chroma(stem, times):
    """Chroma (12, n_steps) of a stem, averaged over each sixteenth; also the per-step energy."""
    import librosa
    import numpy as np
    y, sr = librosa.load(DATA / "stems" / f"{stem}.wav", sr=22050)
    hop = 512
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    edges = np.searchsorted(librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop),
                            np.append(times, times[-1] + (times[-1] - times[-2])))
    out, energy = np.zeros((12, len(times))), np.zeros(len(times))
    for s in range(len(times)):
        a, b = edges[s], max(edges[s + 1], edges[s] + 1)
        out[:, s], energy[s] = C[:, a:b].mean(1), rms[a:b].mean()
    return out, energy


def roll_chroma(classes, events, n_steps):
    """Chroma (12, n_steps) of a transcription: every note sounds for its duration."""
    import numpy as np
    out = np.zeros((12, n_steps))
    for s, c, d, *_ in events:
        out[int(classes[c]) % 12, s:s + d] += 1
    return out


def similarity(classes, events, chroma, energy):
    """Mean cosine similarity between transcription and stem chroma, over the steps where the stem plays."""
    import numpy as np
    R = roll_chroma(classes, events, chroma.shape[1])
    on = energy > np.percentile(energy, 40)
    num = (R * chroma).sum(0)
    den = np.linalg.norm(R, axis=0) * np.linalg.norm(chroma, axis=0) + 1e-9
    return float((num / den)[on].mean())


def score():
    """How close each pitched part of data/song.json is to its stem (chroma similarity, 0-1, octave-blind)."""
    s = json.loads((DATA / "song.json").read_text())
    times = s["times"]
    for p, stem in PART_STEM.items():
        d = s["parts"][p]
        chroma, energy = stem_chroma(d.get("stem", stem), times)
        print(f"  {p:7s} chroma similarity {similarity(d['classes'], d['events'], chroma, energy):.3f}"
              f"  ({len(d['events'])} notes, {len(d['classes'])} classes)")


def song_from_audio(path):
    """-> grid, step_sec, parts, raw notes per part (with the stem they come from, for the samples)."""
    import librosa
    from basic_pitch.inference import predict
    stems = separate(path)
    mix, sr = librosa.load(stems / "mix.wav", sr=22050)
    _, beats = librosa.beat.beat_track(y=mix, sr=sr, units="time")
    grid, step_sec = step_grid(beats)
    heard = grid[grid < len(mix) / sr]  # the sixteenths inside the recording (for the chroma score)

    def bp(stem):
        _, _, ev = predict(str(stems / f"{stem}.wav"), **PITCH[stem])
        return [(s, e, p, a, stem) for s, e, p, a, *_ in ev]

    print("transcribing the stems (basic-pitch, pYIN for mono lines)...")
    other, guitar, piano = bp("other"), bp("guitar"), bp("piano")
    if len(guitar) < MIN_NOTES:
        print(f"guitar stem nearly empty ({len(guitar)} notes): the guitar plays the 'other' stem, the synth the 'piano' one")
        guitar, synth = other, piano
    else:
        synth = piano + other
    candidates = {"bass": {"basic-pitch": bp("bass")}, "guitar": {"basic-pitch": guitar},
                  "synth": {"basic-pitch": synth}, "vocals": {"basic-pitch": bp("vocals")}}
    for part, (fmin, fmax) in PYIN_RANGE.items():
        candidates[part]["pyin"] = [(*n, part) for n in mono_notes(part, fmin, fmax)]
    drums, _ = librosa.load(stems / "drums.wav", sr=sr)
    parts = {"drums": {"classes": list(DRUM_BANDS), "events": drum_events(drums, sr, grid), "stem": "drums"}}
    raw = {}
    for part, cands in candidates.items():
        chroma, energy = stem_chroma(PART_STEM[part], heard)
        best = None
        for method, ns in cands.items():
            classes, events = to_events(ns, grid, step_sec, MONO.get(part))
            sim = similarity(classes, [e for e in events if e[0] < len(heard)], chroma, energy)
            print(f"  {part:7s} {method:11s} chroma similarity {sim:.3f}")
            if best is None or sim > best[0]:
                best = (sim, method, classes, events, ns)
        sim, method, classes, events, raw[part] = best
        parts[part] = {"classes": classes, "events": events, "stem": PART_STEM[part], "method": method,
                       "similarity": round(sim, 3)}
    return grid, step_sec, parts, raw


def cut(y, sr, start, length):
    """A sample from a mono signal: 10 ms before the attack, short fade-in/out, normalized loudness."""
    import numpy as np
    a = max(0, int((start - 0.01) * sr))
    seg = y[a:a + int(length * sr)].copy()
    n_in, n_out = int(0.003 * sr), min(len(seg) // 3, int(0.06 * sr))
    seg[:n_in] *= np.linspace(0, 1, n_in)
    seg[len(seg) - n_out:] *= np.linspace(1, 0, n_out)
    rms = np.sqrt((seg ** 2).mean()) + 1e-9
    seg *= min(0.1 / rms, 0.95 / (np.abs(seg).max() + 1e-9))  # ~-20 dBFS rms, never clipping
    return seg


def samples(raw=None):
    """Cut one sample per note class from each part's own stem (data/samples/<part>/<class>.wav) and
    store the manifest in song.json. The flies then play with the song's real voice, bass, drums..."""
    import librosa
    import numpy as np
    import soundfile as sf
    raw = raw or json.loads((DATA / "notes.json").read_text())
    song_path = DATA / "song.json"
    s = json.loads(song_path.read_text())
    out = DATA / "samples"
    sr, audio, manifest = 44100, {}, {}

    def stem(name):
        if name not in audio:
            audio[name] = librosa.load(DATA / "stems" / f"{name}.wav", sr=sr, mono=True)[0]
        return audio[name]

    for part, notes in raw.items():
        manifest[part] = {}
        (out / part).mkdir(parents=True, exist_ok=True)
        for pitch in s["parts"][part]["classes"]:
            cands = [n for n in notes if int(n[2]) == pitch]
            if not cands:
                continue

            def quality(n):  # loud, long enough, and not covered by other notes of the same part
                start, end, _, amp, _ = n
                overlaps = sum(1 for m in notes if m is not n and m[0] < end and m[1] > start)
                return amp * min(end - start, 1.0) / (1 + overlaps)
            start, end, _, _, src = max(cands, key=quality)
            length = float(np.clip(end - start, *SAMPLE_LEN))
            sf.write(out / part / f"{pitch}.wav", cut(stem(src), sr, start, length), sr, subtype="PCM_16")
            manifest[part][str(pitch)] = f"samples/{part}/{pitch}.wav"
    # drums: the cleanest hit of each piece = loud in its band, quiet in the others
    y = stem("drums")
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    flux = np.array([np.maximum(0, np.diff(S[(freqs >= lo) & (freqs < hi)].sum(0), prepend=0))
                     for lo, hi in DRUM_BANDS.values()])
    flux /= np.percentile(flux, 99.5, axis=1, keepdims=True) + 1e-9
    manifest["drums"] = {}
    (out / "drums").mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(DRUM_BANDS):
        clean = flux[c] - flux[np.arange(len(DRUM_BANDS)) != c].max(0)
        f = int(np.argmax(np.where(flux[c] > 0.5, clean, -np.inf)))
        hit = cut(y, sr, float(librosa.frames_to_time(f, sr=sr, hop_length=512)), DRUM_SAMPLE_LEN[name])
        sf.write(out / "drums" / f"{name}.wav", hit, sr, subtype="PCM_16")
        manifest["drums"][name] = f"samples/drums/{name}.wav"
    s["samples"] = manifest
    song_path.write_text(json.dumps(s))
    print("samples:", {p: len(m) for p, m in manifest.items()}, "->", out)


def song_from_midi(path):
    import numpy as np
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    grid, step_sec = step_grid(pm.get_beats())
    raw = {p: [] for p in PARTS}
    kit = {"kick": {35, 36}, "snare": {37, 38, 39, 40}}
    for inst in pm.instruments:
        name = inst.name.lower()
        if inst.is_drum:
            for n in inst.notes:
                c = next((k for k, ps in kit.items() if n.pitch in ps), "hat" if n.pitch in {42, 44, 46} else None)
                if c:
                    raw["drums"].append((int(np.argmin(np.abs(grid - n.start))), list(DRUM_BANDS).index(c), 1,
                                         round(n.velocity / 127, 2)))
            continue
        p = ("bass" if 32 <= inst.program <= 39 or "bass" in name else
             "guitar" if 24 <= inst.program <= 31 or "guit" in name else
             "vocals" if any(k in name for k in ("voc", "voice", "melod", "lead")) or 52 <= inst.program <= 54 else
             "synth")
        raw[p] += [(n.start, n.end, n.pitch, n.velocity / 127) for n in inst.notes]
    parts = {"drums": {"classes": list(DRUM_BANDS), "events": sorted(set(raw.pop("drums")))}}
    for p, ns in raw.items():
        classes, events = to_events(ns, grid, step_sec, MONO.get(p))
        parts[p] = {"classes": classes, "events": events}
    return grid, step_sec, parts, None


def song(path):
    path = Path(path)
    midi = path.suffix.lower() in (".mid", ".midi")
    grid, step_sec, parts, raw = (song_from_midi if midi else song_from_audio)(path)
    n_steps = 1 + max(e[0] for p in parts.values() for e in p["events"])
    out = {"bpm": round(60 / (step_sec * 4), 1), "step_sec": step_sec, "n_steps": n_steps, "parts": parts,
           "times": [round(float(t), 4) for t in grid[:n_steps]]}  # time (s) in the recording of each sixteenth
    if not midi:
        out["audio"] = "stems/mix.wav"  # relative to data/: the stage plays it as "Original"
    (DATA / "song.json").write_text(json.dumps(out, default=float))
    if raw:  # audio: cut the instrument samples from the stems (kept in notes.json to re-cut with `samples`)
        (DATA / "notes.json").write_text(json.dumps(raw, default=float))  # numpy scalars -> float
        samples(raw)
    print(f"song.json: {out['bpm']} bpm, {n_steps} sixteenths ({n_steps // 16} bars)")
    for p, d in parts.items():
        print(f"  {p:7s} {len(d['events']):5d} notes, {len(d['classes'])} classes")
        if len(d["events"]) < MIN_NOTES:
            print(f"  WARNING: {p} nearly empty")

FLYBODY = "https://raw.githubusercontent.com/TuragaLab/flybody/main/"
FLY_TRIS = 150_000  # triangle budget for one fly (the stage shows five)


def fly():
    """flybody (Janelia + Google DeepMind, Apache-2.0) MuJoCo fly -> data/fly.glb: one glTF node per <body>, named
    like it, with its local transform and its hinge joints in the node extras; visual meshes only, one child mesh
    node per material ("head-red")."""
    import struct
    import xml.etree.ElementTree as ET
    import numpy as np
    import trimesh
    from trimesh.transformations import quaternion_matrix
    src = DATA / "flybody"
    src.mkdir(parents=True, exist_ok=True)

    def get(url, path):
        if not path.exists():
            print("downloading", path.name)
            urllib.request.urlretrieve(url, path)
    get(FLYBODY + "flybody/fruitfly/assets/fruitfly.xml", src / "fruitfly.xml")
    get(FLYBODY + "LICENSE", DATA / "flybody-LICENSE")
    root = ET.parse(src / "fruitfly.xml").getroot()
    assert root.find("compiler").get("angle") == "radian"
    defaults = {}  # class -> {tag: attrs}, inherited down the <default> tree

    def walk(el, parent):
        cls = {t: dict(a) for t, a in parent.items()}
        for c in el:
            if c.tag != "default":
                cls.setdefault(c.tag, {}).update(c.attrib)
        defaults[el.get("class", "main")] = cls
        for c in el.findall("default"):
            walk(c, cls)
    walk(root.find("default"), {})
    attrs = lambda el, cls: {**defaults[el.get("class", cls)].get(el.tag, {}), **el.attrib}
    vec = lambda s, d: np.array([float(x) for x in (s or d).split()])

    def matrix(a):  # MuJoCo pos + quat (w, x, y, z), same order as trimesh
        q = vec(a.get("quat"), "1 0 0 0")
        m = quaternion_matrix(q / np.linalg.norm(q))
        m[:3, 3] = vec(a.get("pos"), "0 0 0")
        return m

    asset = root.find("asset")
    rgba = {m.get("name"): vec(m.get("rgba"), "1 1 1 1") for m in asset.findall("material")}
    meshes = {m.get("name"): attrs(m, "main") for m in asset.findall("mesh")}
    for m in meshes.values():
        get(FLYBODY + "flybody/fruitfly/assets/" + m["file"], src / m["file"])
    mats = {name: trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=(np.clip(c, 0, 1) * 255).round().astype(np.uint8), metallicFactor=0.0,
        roughnessFactor=0.3 if c[3] < 1 else 0.55, alphaMode="BLEND" if c[3] < 1 else "OPAQUE", doubleSided=bool(c[3] < 1))
        for name, c in rgba.items()}
    scene, bodies, parts = trimesh.Scene(), [], []  # parts: (body, material, mesh in the body frame)

    def body(el, parent, cls):
        cls, name = el.get("childclass", cls), el.get("name")
        bodies.append(name)
        joints = []
        for j in el.findall("joint"):
            a = attrs(j, cls)
            if a.get("type", "hinge") == "hinge":
                joints.append({"name": a["name"], "axis": vec(a.get("axis"), "0 0 1").tolist(),
                               "pos": vec(a.get("pos"), "0 0 0").tolist(), "range": vec(a.get("range"), "0 0").tolist(),
                               "springref": float(a.get("springref", 0))})
        scene.graph.update(frame_to=name, frame_from=parent, matrix=matrix(el.attrib),
                           metadata={"joints": joints} if joints else None)
        by_mat = {}
        for g in el.findall("geom"):
            a = attrs(g, cls)
            if "mesh" not in a or a.get("type") != "mesh" or a.get("group") in ("3", "4", "5"):
                continue  # collision capsules/ellipsoids, fluid ellipsoids, inertial boxes
            assert "euler" not in a and "fromto" not in a, a["name"]
            m = meshes[a["mesh"]]
            tm = trimesh.load(src / m["file"], force="mesh")
            tm.apply_scale(vec(m.get("scale"), "1 1 1"))
            tm.apply_transform(matrix(a))
            by_mat.setdefault(a.get("material", "body"), []).append(tm)
        parts.extend((name, mat, trimesh.util.concatenate(ms)) for mat, ms in by_mat.items())
        for c in el.findall("body"):
            body(c, name, cls)
    for b in root.find("worldbody").findall("body"):
        body(b, scene.graph.base_frame, "main")
    tris = sum(len(m.faces) for *_, m in parts)
    if tris > FLY_TRIS:  # quadric decimation of the bulky surfaces only: thin bristles, veins and claws would break
        bulk = lambda mat, m: mat in ("body", "red", "lower", "ocelli") and len(m.faces) > 400
        keep = 1 - (tris - FLY_TRIS) / sum(len(m.faces) for _, mat, m in parts if bulk(mat, m))
        parts = [(b, mat, m.simplify_quadric_decimation(percent=1 - keep) if bulk(mat, m) else m) for b, mat, m in parts]
        print(f"decimated {tris} -> {sum(len(m.faces) for *_, m in parts)} triangles")
    for b, mat, m in parts:
        m.update_faces(m.nondegenerate_faces())  # zero-area faces -> zero normals -> NaN pixels in the browser
        m.remove_unreferenced_vertices()
        nrm = m.vertex_normals.copy()  # a few vertices between back-to-back faces still get none: any unit vector will do
        nrm[np.linalg.norm(nrm, axis=1) < 0.5] = (0, 0, 1)
        m.vertex_normals = nrm
        m.visual = trimesh.visual.TextureVisuals(material=mats[mat])
        scene.add_geometry(m, node_name=f"{b}-{mat}", geom_name=f"{b}-{mat}", parent_node_name=b)
    out = DATA / "fly.glb"
    scene.export(out, include_normals=True)
    # check from the file: every <body> is a node with its name, joints in the extras
    raw = out.read_bytes()
    gltf = json.loads(raw[20:20 + struct.unpack("<I", raw[12:16])[0]])
    nodes = {n["name"]: n for n in gltf["nodes"]}
    assert set(bodies) <= nodes.keys() and len(nodes) == len(bodies) + len(parts), "body nodes missing"
    assert all("joints" in nodes[b].get("extras", {}) for b in bodies if b != "thorax"), "joint extras missing"
    assert all("NORMAL" in p["attributes"] for m in gltf["meshes"] for p in m["primitives"]), "normals missing"
    print(f"fly.glb: {len(bodies)} body nodes + {len(parts)} mesh nodes, {sum(len(m.faces) for *_, m in parts)} triangles, "
          f"{len(raw) / 1e6:.1f} MB, {sum(len(nodes[b]['extras']['joints']) for b in bodies if 'extras' in nodes[b])} joints")


if __name__ == "__main__":
    {"fetch": fetch, "check": check, "score": score, "samples": samples, "fly": fly,
     "song": lambda: song(sys.argv[2])}[sys.argv[1]]()
