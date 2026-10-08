"""Prepare the data for the fly band. Every song lives in data/songs/<name>/ (stems, notes, samples, what the
band learned), so songs never overwrite each other; the connectome and the fly model are shared in data/.

  uv run prep.py fetch                     # FlyWire v783 connectome + annotations
  uv run prep.py fly                       # flybody fly model -> data/fly.glb
  uv run prep.py song path/to/song.mp3 [name]   # new song: stems -> notes -> samples, in data/songs/<name>/
  uv run prep.py song path/to/song.mid [name]   # or from a multitrack MIDI (cleaner targets)
  uv run prep.py song [name]               # redo an existing song from its cached stems (new transcription)
  uv run prep.py list                      # the songs, prepared / trained
  uv run prep.py check [name]              # are the stems complete and consistent with the mix
  uv run prep.py score [name]              # how close each transcribed part is to its stem, per 16-bar section
  uv run prep.py samples [name]            # re-cut the instrument samples from the stems

Without [name] the command asks which song. Everything but fetch/list needs the prep extra: uv sync --extra prep
"""
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
SONGS = DATA / "songs"
MODELS = DATA / "models"  # separation / pitch models, downloaded once and shared by every song
SONG = None  # the song directory the per-song commands work on (set by use())
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
DEMUCS = {"drums": "htdemucs_ft", "bass": "htdemucs_ft",  # ft: better separation
          "guitar": "htdemucs_6s", "piano": "htdemucs_6s", "other": "htdemucs_6s"}  # 6s: the only one with guitar/piano
ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # vocals: ~3 dB SDR better than demucs (Mel/BS-RoFormer papers)
STEMS = [*DEMUCS, "vocals"]
# basic-pitch per stem: instrument range (Hz) and higher thresholds to cut bleed from the other stems
PITCH = {"bass": dict(minimum_frequency=30, maximum_frequency=400, onset_threshold=0.6),
         "vocals": dict(minimum_frequency=80, maximum_frequency=1100, onset_threshold=0.6, frame_threshold=0.4),
         "guitar": dict(minimum_frequency=70, maximum_frequency=1500, onset_threshold=0.5, frame_threshold=0.3),
         "piano": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.5, frame_threshold=0.3),
         "other": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.5, frame_threshold=0.3)}
# (thresholds picked by `score`: synth 0.744 -> 0.801 with these + MAX_POLY 4; vocals 0.496 -> 0.534 with pYIN)
F0_RANGE = {"bass": (30, 400), "vocals": (80, 1100)}  # mono parts that also try f0 trackers; the best transcription wins
MIN_CLASS_NOTES = 3  # notes appearing fewer times than this are almost always transcription errors
MAX_POLY = 4  # max simultaneous notes for the synth (keeps the loudest)
SAMPLE_LEN = (0.3, 1.5)  # seconds: shortest/longest sample cut from a stem
DRUM_SAMPLE_LEN = {"kick": 0.45, "snare": 0.35, "hat": 0.12}
MONO_MIN_SEC = 0.07  # f0 notes shorter than this are glitches
BEAT_HOP = 256  # beat tracking resolution: 11.6 ms at 22050 Hz (512 gave 23 ms steps -> sixteenths jumping 116/122/128 ms)
BEAT_SMOOTH = 8  # beats on each side of the local straight-line fit that removes the remaining jitter
PART_STEM = {"bass": "bass", "guitar": "guitar", "synth": "other", "vocals": "vocals"}  # stem each pitched part is scored against


# ---------- songs ----------
def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "song"


def songs():
    """The song directories, oldest first."""
    return sorted((d for d in SONGS.glob("*") if d.is_dir()), key=lambda d: d.stat().st_ctime) if SONGS.exists() else []


def status(d):
    """One line about a song: prepared? trained? how well?"""
    s = d / "song.json"
    if not s.exists():
        return "not prepared"
    song = json.loads(s.read_text())
    out = f"{song['bpm']} bpm, {song['n_steps'] // 16} bars"
    log = d / "train-log.json"
    if (d / "readout.pt").exists():
        q = json.loads(log.read_text()).get("best_quality") if log.exists() else None
        out += f", trained (quality {q:.4f})" if q else ", trained live"
    else:
        out += ", not trained"
    return out


def pick(name=None, need="song.json"):
    """The song called `name` (or a unique prefix of it); without a name, ask in the terminal."""
    have = [d for d in songs() if (d / need).exists()]
    if name:
        hit = [d for d in have if d.name == name] or [d for d in have if d.name.startswith(name)]
        if len(hit) != 1:
            sys.exit(f"no single song matches '{name}'; songs: {', '.join(d.name for d in have) or 'none'}")
        return hit[0]
    if not have:
        sys.exit("no songs yet: uv run prep.py song path/to/song.mp3")
    if len(have) == 1:
        return have[0]
    for i, d in enumerate(have, 1):
        print(f"  {i}. {d.name:30s} {status(d)}")
    while True:
        a = input(f"which song? [1-{len(have)}] ").strip()
        if a.isdigit() and 1 <= int(a) <= len(have):
            return have[int(a) - 1]


def use(d):
    global SONG
    SONG = Path(d)
    return SONG


def list_songs():
    for d in songs():
        print(f"  {d.name:30s} {status(d)}")
    if not songs():
        print("  no songs yet: uv run prep.py song path/to/song.mp3")


def fetch():
    DATA.mkdir(exist_ok=True)
    for name, url in URLS.items():
        if not (DATA / name).exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, DATA / name)
    print("data ok")


# ---------- timing ----------
def smooth_beats(beats):
    """Each beat replaced by a straight-line fit through its neighbours: removes the frame jitter of the tracker
    but still follows real tempo changes."""
    import numpy as np
    beats = np.asarray(beats, float)
    k, out = np.arange(len(beats)), np.empty(len(beats))
    for i in range(len(beats)):
        a, b = max(0, i - BEAT_SMOOTH), min(len(beats), i + BEAT_SMOOTH + 1)
        out[i] = np.polyval(np.polyfit(k[a:b], beats[a:b], 1), i)
    return out


def step_grid(beats, n_after=64):
    """Sixteenth times: each interval between beats split in 4 (follows the song's tempo). Also the mean sixteenth."""
    import numpy as np
    beats = np.asarray(beats, float)
    beat = float(np.mean(np.diff(beats)))
    head = beats[0] - beat * np.arange(int(beats[0] // beat), 0, -1)
    tail = beats[-1] + beat * np.arange(1, n_after)
    beats = np.concatenate([head, beats, tail])
    t = (beats[:-1, None] + np.diff(beats)[:, None] * np.arange(4) / 4).ravel()
    return t, beat / 4


# ---------- transcription ----------
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


def roformer_vocals(mix, out):
    """Vocals with BS-RoFormer (audio-separator, UVR models), same length/rate as the mix."""
    import os
    import imageio_ffmpeg
    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator
    # audio-separator insists on an `ffmpeg` on the PATH: use the one bundled with imageio-ffmpeg, renamed
    ffmpeg = MODELS / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg.exists():
        ffmpeg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    os.environ["PATH"] = f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}"
    tmp = SONG / "stems" / "roformer"
    sep = Separator(output_dir=str(tmp), model_file_dir=str(MODELS), output_single_stem="Vocals")
    sep.load_model(ROFORMER)
    print("BS-RoFormer vocals (first run downloads the model)...")
    files = [Path(f) if Path(f).is_absolute() else tmp / f for f in sep.separate(str(mix))]
    y, sr = sf.read(files[0])
    ref, rsr = sf.read(mix)
    assert sr == rsr, f"RoFormer wrote {sr} Hz, the mix is {rsr} Hz"
    y = np.pad(y, ((0, max(0, len(ref) - len(y))), (0, 0)))[: len(ref)]
    sf.write(out, y, sr)
    shutil.rmtree(tmp, ignore_errors=True)


def separate(path):
    """audio -> stem wavs in <song>/stems (cached): demucs for the band, BS-RoFormer for the vocals.
    IO via librosa/soundfile: no ffmpeg/torchaudio."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    out = SONG / "stems"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "mix.wav").exists():
        y, sr = librosa.load(path, sr=44100, mono=False)
        sf.write(out / "mix.wav", (np.stack([y, y]) if y.ndim == 1 else y).T, sr)
    if not all((out / f"{n}.wav").exists() for n in DEMUCS):
        y, sr = sf.read(out / "mix.wav")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for name in dict.fromkeys(DEMUCS.values()):
            model = get_model(name).eval()
            x = torch.tensor(y.T, dtype=torch.float32)[None]
            ref = x.mean(1)
            print(f"demucs {name} on {dev} (a few minutes)...")
            with torch.no_grad():  # shifts=2: average over 2 time shifts, cleaner separation
                stems = apply_model(model, (x - ref.mean()) / ref.std(), device=dev, shifts=2, split=True,
                                    overlap=0.25, progress=True)[0]
            stems = stems * ref.std() + ref.mean()
            for stem, s in zip(model.sources, stems.numpy()):
                if DEMUCS.get(stem) == name:
                    sf.write(out / f"{stem}.wav", s.T, sr)
    if not (out / "vocals.roformer").exists():  # the marker: older songs have demucs vocals here
        roformer_vocals(out / "mix.wav", out / "vocals.wav")
        (out / "vocals.roformer").touch()
    check()
    return out


def check():
    """Stems complete (same length as the mix), not silent, and how well their sum rebuilds the mix."""
    import numpy as np
    import soundfile as sf
    out = SONG / "stems"
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
    print(f"residual mix - stem sum: {res:.1f} dB (rough: the stems come from three models, they don't sum exactly to the mix)")


def f0_to_notes(t, midi, rms, onsets):
    """A note per run of the same semitone (midi < 0 = unvoiced), split at onsets. -> [(start, end, pitch, amp)]."""
    import numpy as np
    import scipy.ndimage
    midi = scipy.ndimage.median_filter(midi, 5)  # drop single-frame octave/semitone flickers
    min_frames = max(2, int(MONO_MIN_SEC / (t[1] - t[0])))
    notes, start = [], 0
    for f in range(1, len(midi) + 1):
        if f == len(midi) or midi[f] != midi[start] or f in onsets:
            if midi[start] > 0 and f - start >= min_frames:
                notes.append((float(t[start]), float(t[min(f, len(t) - 1)]), int(midi[start]), float(np.mean(rms[start:f]))))
            start = f
    return notes


def pyin_notes(stem, fmin, fmax):
    import librosa
    import numpy as np
    y, sr = librosa.load(SONG / "stems" / f"{stem}.wav", sr=22050)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop)
    midi = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][: len(midi)]
    onsets = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True))
    return f0_to_notes(librosa.frames_to_time(np.arange(len(midi) + 1), sr=sr, hop_length=hop), midi, rms, onsets)


def rmvpe_notes(stem, fmin, fmax, threshold=0.03):  # 0.03 as in RVC; 0.5 drops most sung notes (vocals 0.41 vs 0.55)
    """RMVPE (Wei et al. 2023): f0 tracker built for vocals inside polyphonic music."""
    import librosa
    import numpy as np
    from rmvpe_onnx import RMVPE
    y, sr = librosa.load(SONG / "stems" / f"{stem}.wav", sr=16000)
    t, f0, conf, _ = RMVPE(model_path=MODELS / "rmvpe.onnx").predict(y, sr)
    ok = (conf > threshold) & (f0 >= fmin) & (f0 <= fmax)
    midi = np.where(ok, np.round(librosa.hz_to_midi(np.maximum(f0, 1.0))), -1)
    hop = 160  # RMVPE frames: 10 ms at 16 kHz
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    rms = np.pad(rms, (0, max(0, len(midi) - len(rms))))[: len(midi)]
    onsets = set(librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True))
    return f0_to_notes(np.append(t, t[-1] + hop / sr), midi, rms, onsets)


def stem_chroma(stem, times):
    """Chroma (12, n_steps) of a stem, averaged over each sixteenth; also the per-step energy."""
    import librosa
    import numpy as np
    y, sr = librosa.load(SONG / "stems" / f"{stem}.wav", sr=22050)
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
        if s < n_steps:
            out[int(classes[c]) % 12, s:s + d] += 1
    return out


def similarity(classes, events, chroma, energy, steps=None):
    """Mean cosine similarity between transcription and stem chroma, over the steps where the stem plays
    (optionally only within `steps`, a slice)."""
    import numpy as np
    R = roll_chroma(classes, events, chroma.shape[1])
    on = energy > np.percentile(energy, 40)
    cos = (R * chroma).sum(0) / (np.linalg.norm(R, axis=0) * np.linalg.norm(chroma, axis=0) + 1e-9)
    if steps is not None:
        cos, on = cos[steps], on[steps]
    return float(cos[on].mean()) if on.any() else float("nan")


def score():
    """How close each pitched part is to its stem (chroma similarity, 0-1, octave-blind): whole song and per 16 bars."""
    import numpy as np
    s = json.loads((SONG / "song.json").read_text())
    times = np.array(s["times"])
    heard = int((times < s.get("duration", times[-1] + 1)).sum())
    sims = {}
    for p, stem in PART_STEM.items():
        d = s["parts"][p]
        chroma, energy = stem_chroma(d.get("stem", stem), times[:heard])
        sims[p] = (d, chroma, energy)
        print(f"  {p:7s} chroma similarity {similarity(d['classes'], d['events'], chroma, energy):.3f}"
              f"  ({len(d['events'])} notes, {len(d['classes'])} classes, {d.get('method', '?')})")
    print("  per 16 bars:  " + "  ".join(f"{p:>7s}" for p in PART_STEM))
    for a in range(0, heard, 256):
        sl = slice(a, min(a + 256, heard))
        print(f"  {a // 16 + 1:4d}-{sl.stop // 16:<4d}     " + "  ".join(
            f"{similarity(d['classes'], d['events'], C, E, sl):7.2f}" for d, C, E in sims.values()))


SECTION = 256  # sixteenths (16 bars): each section of a part takes the transcription closest to its stem there


def splice(made, chroma, energy, grid):
    """made: {method: (classes, events, raw notes)}. Per section, the method with the best chroma similarity.
    -> classes, events, raw notes (those whose onset falls in a section taken from their method), plan text."""
    import math
    n = chroma.shape[1]
    choice = []
    for a in range(0, n, SECTION):
        sl = slice(a, min(a + SECTION, n))
        sims = {m: similarity(c, e, chroma, energy, sl) for m, (c, e, _) in made.items()}
        choice.append(max(sims, key=lambda m: -1 if math.isnan(sims[m]) else sims[m]))
    pitched = [(s, cls[c], d, *v) for m, (cls, ev, _) in made.items() for s, c, d, *v in ev if choice[s // SECTION] == m]
    classes = sorted({p for _, p, *_ in pitched})
    idx = {p: i for i, p in enumerate(classes)}
    events = sorted([s, idx[p], d, *v] for s, p, d, *v in pitched)
    edges = list(grid[::SECTION][: len(choice)]) + [math.inf]
    raw = [nt for k, m in enumerate(choice) for nt in made[m][2] if edges[k] <= nt[0] < edges[k + 1]]
    counts = {m: choice.count(m) for m in dict.fromkeys(choice)}
    return classes, events, raw, ", ".join(f"{m} x{k}" for m, k in counts.items())


def song_from_audio(path):
    """-> grid, step_sec, parts, raw notes per part (with the stem they come from, for the samples)."""
    import librosa
    from basic_pitch.inference import predict
    stems = separate(path)
    mix, sr = librosa.load(stems / "mix.wav", sr=22050)
    _, beats = librosa.beat.beat_track(y=mix, sr=sr, hop_length=BEAT_HOP, units="time")
    grid, step_sec = step_grid(smooth_beats(beats))
    heard = grid[grid < len(mix) / sr]  # the sixteenths inside the recording (for the chroma score)

    def bp(stem):
        _, _, ev = predict(str(stems / f"{stem}.wav"), **PITCH[stem])
        return [(s, e, p, a, stem) for s, e, p, a, *_ in ev]

    print("transcribing the stems (basic-pitch; pYIN and RMVPE for mono lines)...")
    other, guitar, piano = bp("other"), bp("guitar"), bp("piano")
    if len(guitar) < MIN_NOTES:
        print(f"guitar stem nearly empty ({len(guitar)} notes): the guitar plays the 'other' stem, the synth the 'piano' one")
        guitar, synth = other, piano
        guitar_stem = "other"
    else:
        synth, guitar_stem = piano + other, "guitar"
    candidates = {"bass": {"basic-pitch": bp("bass")}, "guitar": {"basic-pitch": guitar},
                  "synth": {"basic-pitch": synth}, "vocals": {"basic-pitch": bp("vocals")}}
    for part, (fmin, fmax) in F0_RANGE.items():
        candidates[part]["pyin"] = [(*n, part) for n in pyin_notes(part, fmin, fmax)]
        candidates[part]["rmvpe"] = [(*n, part) for n in rmvpe_notes(part, fmin, fmax)]
    drums, _ = librosa.load(stems / "drums.wav", sr=sr)
    parts = {"drums": {"classes": list(DRUM_BANDS), "events": drum_events(drums, sr, grid), "stem": "drums"}}
    raw = {}
    for part, cands in candidates.items():
        chroma, energy = stem_chroma(PART_STEM[part], heard)
        made = {}
        for method, ns in cands.items():
            classes, events = to_events(ns, grid, step_sec, MONO.get(part))
            made[method] = (classes, [e for e in events if e[0] < len(heard)], ns)
            print(f"  {part:7s} {method:11s} chroma similarity {similarity(*made[method][:2], chroma, energy):.3f}")
        classes, events, raw[part], plan = splice(made, chroma, energy, grid)
        sim = similarity(classes, events, chroma, energy)
        print(f"  {part:7s} {'spliced':11s} chroma similarity {sim:.3f}  ({plan})")
        parts[part] = {"classes": classes, "events": events, "stem": PART_STEM[part], "method": plan,
                       "similarity": round(sim, 3)}
    parts["guitar"]["play_stem"] = guitar_stem  # what the stage plays back for the guitar in stem mode
    return grid, step_sec, parts, raw, len(mix) / sr


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
    """Cut one sample per note class from each part's own stem (<song>/samples/<part>/<class>.wav) and
    store the manifest in song.json. The flies then play with the song's real voice, bass, drums..."""
    import librosa
    import numpy as np
    import soundfile as sf
    raw = raw or json.loads((SONG / "notes.json").read_text())
    song_path = SONG / "song.json"
    s = json.loads(song_path.read_text())
    out = SONG / "samples"
    sr, audio, manifest = 44100, {}, {}

    def stem(name):
        if name not in audio:
            audio[name] = librosa.load(SONG / "stems" / f"{name}.wav", sr=sr, mono=True)[0]
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


def web_audio(parts):
    """Compressed mono copies for the stage (<song>/web/*.ogg): the mix ("Original") and, per part, the stem its
    fly plays back in stem mode. The wavs (45 MB each) would be too heavy for the browser."""
    import librosa
    import soundfile as sf
    out = SONG / "web"
    out.mkdir(exist_ok=True)
    stems = {"mix": "mix", "drums": "drums", **{p: parts[p].get("play_stem", parts[p]["stem"]) for p in PART_STEM}}
    for name, stem in stems.items():
        y, sr = librosa.load(SONG / "stems" / f"{stem}.wav", sr=44100, mono=name != "mix")
        y = y.T if y.ndim > 1 else y
        with sf.SoundFile(out / f"{name}.ogg", "w", sr, y.shape[1] if y.ndim > 1 else 1, format="OGG", subtype="VORBIS") as f:
            for a in range(0, len(y), 1 << 15):  # in blocks: libsndfile's Vorbis encoder crashes on one huge write
                f.write(y[a:a + (1 << 15)])
    return {name: f"web/{name}.ogg" for name in stems}


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
    return grid, step_sec, parts, None, pm.get_end_time()


def song(arg=None, name=None):
    """`song file.mp3 [name]`: a new song from a file; `song [name]`: redo an existing one from its source."""
    path = Path(arg) if arg else None
    if path and path.is_file():
        d = use(SONGS / slugify(name or path.stem))
        d.mkdir(parents=True, exist_ok=True)
        src = d / f"source{path.suffix.lower()}"
        if path.resolve() != src.resolve():
            shutil.copy2(path, src)
    else:
        d = use(pick(arg, need="."))
        src = next(d.glob("source.*"), None)
        if src is None:
            sys.exit(f"{d.name}: no source file; add it with uv run prep.py song path/to/file {d.name}")
    midi = src.suffix in (".mid", ".midi")
    grid, step_sec, parts, raw, duration = (song_from_midi if midi else song_from_audio)(src)
    n_steps = 1 + max(e[0] for p in parts.values() for e in p["events"])
    old = json.loads((d / "song.json").read_text()) if (d / "song.json").exists() else {}
    out = {"title": old.get("title", d.name.replace("-", " ").title()), "bpm": round(60 / (step_sec * 4), 1),
           "step_sec": step_sec, "n_steps": n_steps, "duration": round(duration, 3), "parts": parts,
           "times": [round(float(t), 4) for t in grid[:n_steps]]}  # time (s) in the recording of each sixteenth
    if not midi:
        out["audio"] = "web/mix.ogg"  # relative to the song dir: the stage plays it as "Original"
        out["stems"] = web_audio(parts)
    (d / "song.json").write_text(json.dumps(out, default=float))
    if raw:  # audio: cut the instrument samples from the stems (kept in notes.json to re-cut with `samples`)
        (d / "notes.json").write_text(json.dumps(raw, default=float))  # numpy scalars -> float
        samples(raw)
    print(f"{d.name}: {out['bpm']} bpm, {n_steps} sixteenths ({n_steps // 16} bars)")
    for p, dd in parts.items():
        print(f"  {p:7s} {len(dd['events']):5d} notes, {len(dd['classes'])} classes")
        if len(dd["events"]) < MIN_NOTES:
            print(f"  WARNING: {p} nearly empty")
    if (d / "readout.pt").exists():  # the new notes make that training stale: keep it aside, never mixed up with them
        for f in ("readout.pt", "train-log.json"):
            if (d / f).exists():
                (d / f).replace(d / f.replace(".", ".old.", 1))
        print(f"  its old training no longer matches the new notes (archived as readout.old.pt): uv run brain.py train {d.name}")

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
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    if cmd == "fetch":
        fetch()
    elif cmd == "fly":
        fly()
    elif cmd == "list":
        list_songs()
    elif cmd == "song":
        song(*args[:2])
    elif cmd in ("check", "score", "samples"):
        use(pick(args[0] if args else None, need="stems/mix.wav" if cmd == "check" else "song.json"))
        {"check": check, "score": score, "samples": samples}[cmd]()
    else:
        print(__doc__)
