"""Prepare the data for the fly band.

  uv run prep.py fetch                 # FlyWire v783 connectome + annotations
  uv run prep.py song song.mp3         # mp3 -> stems (demucs) -> notes (basic-pitch) -> data/song.json
  uv run prep.py song song.mid         # or from a multitrack MIDI (cleaner targets)
  uv run prep.py check                 # check that the stems in data/stems are complete and consistent with the mix

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
         "guitar": dict(minimum_frequency=70, maximum_frequency=1500, onset_threshold=0.6),
         "piano": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.6, frame_threshold=0.4),
         "other": dict(minimum_frequency=60, maximum_frequency=3000, onset_threshold=0.6, frame_threshold=0.4)}
MIN_CLASS_NOTES = 3  # notes appearing fewer times than this are almost always transcription errors
MAX_POLY = 3  # max simultaneous notes for the synth (keeps the loudest)


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
    """notes: [(start, end, pitch, amplitude)] -> classes (sorted pitches) and events [step, class, duration_in_steps]."""
    from collections import Counter
    import numpy as np
    common = {p for p, n in Counter(int(n[2]) for n in notes).items() if n >= MIN_CLASS_NOTES}
    by_step, amp = {}, {}
    for start, end, pitch, a in notes:
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
    events = [[s, idx[p], d] for s in sorted(by_step) for p, d in by_step[s].items()]
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
        floor = DRUM_GATE * np.percentile(flux, 99)
        for f in librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=False):
            if flux[max(0, f - 2):f + 3].max() >= floor:  # onset_detect normalizes: drop "hits" that are only bleed from other bands
                events.append((int(np.argmin(np.abs(grid - librosa.frames_to_time(f, sr=sr)))), c, 1))
    return sorted(set(events))


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


def song_from_audio(path):
    import librosa
    from basic_pitch.inference import predict
    stems = separate(path)
    mix, sr = librosa.load(stems / "mix.wav", sr=22050)
    _, beats = librosa.beat.beat_track(y=mix, sr=sr, units="time")
    grid, step_sec = step_grid(beats)

    def notes(stem):
        _, _, ev = predict(str(stems / f"{stem}.wav"), **PITCH[stem])
        return [(s, e, p, a) for s, e, p, a, *_ in ev]

    print("transcribing the stems with basic-pitch...")
    other, guitar, piano = notes("other"), notes("guitar"), notes("piano")
    if len(guitar) < MIN_NOTES:
        print(f"guitar stem nearly empty ({len(guitar)} notes): the guitar plays the 'other' stem, the synth the 'piano' one")
        guitar, synth = other, piano
    else:
        synth = piano + other
    raw = {"bass": notes("bass"), "guitar": guitar, "synth": synth, "vocals": notes("vocals")}
    drums, _ = librosa.load(stems / "drums.wav", sr=sr)
    parts = {"drums": {"classes": list(DRUM_BANDS), "events": drum_events(drums, sr, grid)}}
    for p, ns in raw.items():
        classes, events = to_events(ns, grid, step_sec, MONO.get(p))
        parts[p] = {"classes": classes, "events": events}
    return grid, step_sec, parts


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
                    raw["drums"].append((int(np.argmin(np.abs(grid - n.start))), list(DRUM_BANDS).index(c), 1))
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
    return grid, step_sec, parts


def song(path):
    path = Path(path)
    grid, step_sec, parts = (song_from_midi if path.suffix.lower() in (".mid", ".midi") else song_from_audio)(path)
    n_steps = 1 + max(e[0] for p in parts.values() for e in p["events"])
    out = {"bpm": round(60 / (step_sec * 4), 1), "step_sec": step_sec, "n_steps": n_steps, "parts": parts,
           "times": [round(float(t), 4) for t in grid[:n_steps]]}  # time (s) in the recording of each sixteenth
    if path.suffix.lower() not in (".mid", ".midi"):
        out["audio"] = "stems/mix.wav"  # relative to data/: the stage plays it as "Original"
    (DATA / "song.json").write_text(json.dumps(out))
    print(f"song.json: {out['bpm']} bpm, {n_steps} sixteenths ({n_steps // 16} bars)")
    for p, d in parts.items():
        print(f"  {p:7s} {len(d['events']):5d} notes, {len(d['classes'])} classes")
        if len(d["events"]) < MIN_NOTES:
            print(f"  WARNING: {p} nearly empty")


if __name__ == "__main__":
    {"fetch": fetch, "check": check, "song": lambda: song(sys.argv[2])}[sys.argv[1]]()
