"""Drosophila brains (FlyWire v783 connectome, LIF from Shiu et al. 2024) on GPU, one per instrument.

The connectome is frozen (reservoir): each fly "hears" the position in the song through its
Johnston's organ auditory neurons, and learns only a linear readout from its output neurons
to the notes of its own part.

  uv run brain.py                     # self-check: speed, neuron response, learning over 8 bars
  uv run brain.py train [song] [minutes]   # offline training on a whole song, early stopping
                                           # -> data/songs/<song>/readout.pt (asks which song if not given)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).parent / "data"
PARTS = ["drums", "bass", "guitar", "synth", "vocals"]
MONO = {"bass", "guitar", "vocals"}

# Shiu et al. 2024, model.py (mV, ms, kHz)
V0, VTH, T_MBR, TAU, T_REF, T_DLY, W_SYN, F_POI, R_POI = -52.0, -45.0, 20.0, 5.0, 2.2, 1.8, 0.275, 250, 0.150
DT = 0.5  # ponytail: dt 0.5ms instead of 0.1ms to keep up with real time; lower it if rates drift from the reference
N_DLY, N_REF = round(T_DLY / DT), round(T_REF / DT)
RLS_P0 = 10.0  # RLS: initial covariance (= 1/regularization); picked by `brain.py train` (1.0 and 0.1 scored lower)
N_RASTER = 48  # output neurons (the most active) shown in the stage raster
EARS = ["auditory", "wind_gravity", "grooming"]  # antennal JO neurons: the "sound" input
PROBE_STEPS = 16
BINS = 4  # spike counts in 4 sub-windows per sixteenth: 4x features (rank) at zero simulation cost
MAX_DUR = 16  # longest note the readout predicts, in sixteenths
BRAIN_POINTS = 40000  # neurons drawn as the brain silhouette in the stage map (out of ~139k)


def load_connectome():
    """Return (post, pre, weight_mV) as numpy arrays, plus per-neuron annotations (index = Presynaptic_Index)."""
    ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index
    c = pd.read_parquet(DATA / "Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    ann = (pd.read_csv(DATA / "annotations.tsv", sep="\t",
                       usecols=["root_id", "super_class", "cell_sub_class", "pos_x", "pos_y", "pos_z"],
                       low_memory=False).drop_duplicates("root_id").set_index("root_id").reindex(ids))
    edges = (c.Postsynaptic_Index.values, c.Presynaptic_Index.values, c["Excitatory x Connectivity"].values * W_SYN)
    return edges, ann.reset_index().rename(columns={"index": "root_id"})


def matrix(edges, n, keep=None, dev="cuda"):
    """Synaptic matrix W[post, pre]: sparse over the whole brain, dense over the `keep` sub-circuit."""
    post, pre, w = edges
    if keep is None:
        idx = torch.tensor(np.stack([post, pre]))
        return torch.sparse_coo_tensor(idx, torch.tensor(w, dtype=torch.float32), (n, n)).coalesce().to_sparse_csr().to(dev)
    m = np.isin(post, keep) & np.isin(pre, keep)
    M = torch.zeros(len(keep), len(keep))
    M.index_put_((torch.tensor(np.searchsorted(keep, post[m])), torch.tensor(np.searchsorted(keep, pre[m]))),
                 torch.tensor(w[m], dtype=torch.float32), accumulate=True)
    return M.to(dev)


class Part:
    """A fly's readout: features -> note onsets of its part, plus how loud and how long they are.

    Linear regression learned online with recursive least squares (RLS): at every sixteenth, the exact
    least-squares solution over everything heard so far. Converges in 1-2 rehearsals even on the whole
    song (NLMS gradient descent needed ~30). Two weight blocks share the same P: W picks the notes,
    R predicts velocity per class and the note length. RLS columns are independent, so R never
    changes which notes are played."""

    def __init__(self, name, spec, n_steps, n_feat, dev):
        self.name, self.classes, self.mono = name, spec["classes"], name in MONO
        C = len(self.classes)
        Y = torch.zeros(n_steps, C + self.mono)
        Yr = torch.zeros(n_steps, C + 1)  # velocity of each class at its onset, then the length / MAX_DUR
        self.truth = {}  # step -> {class: (length, velocity)} as in the song
        durs = {}
        for s, c, d, *v in spec["events"]:
            vel = v[0] if v else 0.8
            durs.setdefault(c, []).append(d)
            if s < n_steps:
                Y[s, c], Yr[s, c] = 1, vel
                Yr[s, C] = max(Yr[s, C], min(d, MAX_DUR) / MAX_DUR)
                self.truth.setdefault(s, {})[c] = (d, vel)
        if self.mono:  # extra "no onset" class: the note played is the argmax
            Y[:, C] = Y[:, :C].sum(1) == 0
        self.Y, self.Yr = Y.to(dev), Yr.to(dev)
        self.W = torch.zeros(n_feat + 1, Y.shape[1], device=dev)
        self.R = torch.zeros(n_feat + 1, Yr.shape[1], device=dev)
        self.z = torch.zeros(C + self.mono, device=dev)
        self.reset_learning()
        self.dur = [int(np.median(durs.get(c, [1]))) for c in range(C)]
        self.tp = self.fp = self.fn = 0
        self.vel_err = self.dur_err = self.n_err = 0.0

    def reset_learning(self):
        # ponytail: P (~50 MB in float64) is not saved; on resume it restarts from RLS_P0 with the already learned W
        n = self.W.shape[0]
        self.P = torch.eye(n, dtype=torch.float64, device=self.W.device) * RLS_P0

    def __call__(self, x, s, learn=True):
        """x: spike counts (F+1,) with bias. Return (played, expected) as lists of (class index, length, velocity)."""
        C, y = len(self.classes), self.Y[s]
        z, zr = x @ self.W, x @ self.R
        self.z = z
        if self.mono:
            k = int(z.argmax())
            idx = [] if k == C else [k]
        else:
            idx = (z > 0.5).nonzero().flatten().tolist()
        played = []
        if idx:
            reg = zr.tolist()
            for c in idx:
                length = int(np.clip(round(reg[C] * MAX_DUR), 1, MAX_DUR)) if self.mono else self.dur[c]
                played.append((c, length, round(float(np.clip(reg[c], 0.15, 1.0)), 2)))
        if learn:
            xd = x.double()
            Px = self.P @ xd
            gain = Px / (1 + xd @ Px)
            self.P -= torch.outer(gain, Px)
            g = gain.float()
            self.W += torch.outer(g, y - z)
            self.R += torch.outer(g, self.Yr[s] - zr)
        target = [(c, *self.truth[s][c]) for c in sorted(self.truth.get(s, {}))]
        hit = len(set(idx) & {t[0] for t in target})
        for c, length, vel in played:  # how close loudness and length are, on the right notes
            if c in self.truth.get(s, {}):
                d, v = self.truth[s][c]
                self.vel_err += abs(vel - v)
                self.dur_err += abs(length - min(d, MAX_DUR)) if self.mono else 0.0
                self.n_err += 1
        self.tp, self.fp, self.fn = self.tp + hit, self.fp + len(idx) - hit, self.fn + len(target) - hit
        return played, target

    def top(self, k=5):
        """The k classes the readout scores highest right now: [(class value, score)], with "rest" for no note
        (the mono parts' extra class; for polyphonic parts 1 - the best note score)."""
        C = len(self.classes)
        names = self.classes + ["rest"]
        z = self.z if self.mono else torch.cat([self.z, (1 - self.z.max()).clamp(0, 1).view(1)])
        vals, idx = z.topk(min(k, C + 1))
        return [(names[i], round(float(v), 3)) for v, i in zip(vals.tolist(), idx.tolist())]

    def score(self):
        """F1, precision and recall on onsets since the last call (and reset the counters)."""
        tp, fp, fn = self.tp, self.fp, self.fn
        n, ve, de = max(1, self.n_err), self.vel_err, self.dur_err
        self.tp = self.fp = self.fn = 0
        self.vel_err = self.dur_err = self.n_err = 0.0
        return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                "vel_mae": ve / n, "dur_mae": de / n}

    def f1(self):
        return self.score()["f1"]


class Band:
    def __init__(self, song, dev="cuda"):
        self.song, self.dev, self.B = song, dev, len(PARTS)
        edges, ann = load_connectome()
        self.n_dt = round(song["step_sec"] * 1000 / DT)
        ears = np.flatnonzero(ann.cell_sub_class.isin(EARS))
        # Probe on the whole brain (~10 s): all antennae active -> which neurons can fire.
        # The other ~137k stay below threshold and affect no one: only the live sub-circuit is simulated.
        # ponytail: neurons that fire only with partial codes (disinhibition) are lost; lengthen the probe if needed
        self.N = len(ann)
        self.M = matrix(edges, self.N, dev=dev)
        self.reset()
        torch.manual_seed(0)  # same circuit on every start, so a saved readout.pt stays valid
        all_ears = torch.tensor(ears, device=dev)
        fired = sum(self.run(all_ears).sum((0, 2)) for _ in range(PROBE_STEPS)).cpu().numpy()
        alive = np.union1d(np.flatnonzero(fired > 0), ears)
        self.M, self.N = matrix(edges, self.N, alive, dev), len(alive)
        self.neurons = ann.iloc[alive].reset_index(drop=True)  # root_id / class of each simulated neuron
        self.layout(ann, alive)
        rng = np.random.default_rng(0)
        ears = rng.permutation(np.searchsorted(alive, ears))
        self.ear_pos, self.ear_bar = np.array_split(ears, 2)  # half hears the sixteenth within the bar, half the bar
        self.pos_code = [torch.tensor(rng.choice(self.ear_pos, len(self.ear_pos) // 4, replace=False), device=dev)
                         for _ in range(16)]
        feat = np.setdiff1d(np.arange(self.N), ears)  # the whole live circuit except the inputs
        self.feat = torch.tensor(feat, device=dev)
        # for the stage raster: the most active outputs in the probe (positions within feat)
        self.raster = torch.tensor(np.argsort(-fired[alive[feat]], kind="stable")[:N_RASTER], device=dev)
        self.parts = [Part(p, song["parts"][p], song["n_steps"], BINS * len(feat), dev) for p in PARTS]
        self.reset()

    def layout(self, ann, alive):
        """3D positions for the stage brain map: a silhouette of the whole brain and the simulated neurons.
        FlyWire voxels are 4x4x40 nm, so z is stretched x10; ~600 neurons without annotation have no position."""
        pos = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * [1, 1, 10]
        known = ~np.isnan(pos).any(1)
        center = pos[known].mean(0)
        scale = np.abs(pos[known] - center).max()
        pos = (pos - center) / scale
        rng = np.random.default_rng(0)
        shape = rng.choice(np.flatnonzero(known), min(BRAIN_POINTS, int(known.sum())), replace=False)
        self.shape_pos = pos[shape].astype(np.float32)
        self.alive_pos = np.nan_to_num(pos[alive], nan=0.0).astype(np.float32)  # unknown -> brain centre
        self.alive_class = ann.super_class.iloc[alive].fillna("unknown").tolist()

    def reset(self):
        z = lambda: torch.zeros(self.N, self.B, device=self.dev)
        self.v, self.g, self.ref = z() + V0, z(), z()
        self.buf, self.k = [z() for _ in range(N_DLY)], 0

    def code(self, s):
        """Active antennae at sixteenth s: code for the position within the bar + for the bar."""
        bar = np.random.default_rng(1000 + s // 16).choice(self.ear_bar, len(self.ear_bar) // 4, replace=False)
        return torch.cat([self.pos_code[s % 16], torch.tensor(bar, device=self.dev)])

    def listen(self, s):
        """Simulate the 5 brains for sixteenth s. Return spike counts (B, BINS*F) of the output neurons."""
        counts = self.run(self.code(s), seed=s)
        self.last_counts = counts.sum(0)  # (N, B): every simulated neuron, for the brain map
        return counts[:, self.feat].permute(2, 0, 1).reshape(self.B, -1)

    @torch.no_grad()
    def run(self, ear, seed=None):
        """One sixteenth of simulation with Poisson input on the `ear` antennae. Return spike counts (BINS, N, B).
        With `seed` the input spikes are the same on every rehearsal: the fly hears the same recording again
        (each fly still has its own stream). Without it, fresh Poisson noise."""
        rng = torch.Generator(self.dev).manual_seed(seed) if seed is not None else None
        decay = math.exp(-DT / TAU)
        v, g, ref = self.v, self.g, self.ref
        counts = torch.zeros(BINS, self.N, self.B, device=self.dev)
        for i in range(self.n_dt):
            g += self.buf[self.k]
            g[ear] += (torch.rand(len(ear), self.B, device=self.dev, generator=rng) < R_POI * DT) * (F_POI * W_SYN)
            free = ref <= 0
            v.copy_(torch.where(free, v + DT * (V0 - v + g) / T_MBR, v))
            g.copy_(torch.where(free, g * decay, g))
            spk = v > VTH
            v.masked_fill_(spk, V0)
            g.masked_fill_(spk, 0)
            ref.copy_(torch.where(spk, float(N_REF), ref - 1))
            self.buf[self.k] = self.M @ spk.float()  # arrives after N_DLY steps
            self.k = (self.k + 1) % N_DLY
            counts[i * BINS // self.n_dt] += spk
        return counts

    @torch.no_grad()
    def play(self, r, s, learn=True):
        """r: (B, BINS*F) from listen. Each fly plays (and learns) its part.
        Return {part: (played, expected)}, notes as (class index, length, velocity)."""
        x = torch.cat([r, torch.ones(self.B, 1, device=self.dev)], 1)
        return {p.name: p(x[i], s, learn) for i, p in enumerate(self.parts)}

    def spikes(self, r):
        """Spike counts (B, N_RASTER) of the neurons shown in the raster, from a listen output."""
        return r.view(self.B, BINS, -1).sum(1)[:, self.raster]

    def state(self):
        return {"W": [p.W for p in self.parts], "R": [p.R for p in self.parts], "feat": self.feat}

    def load(self, st):
        """Resume a saved readout only if circuit and classes match. Return whether it was applied."""
        ok = (st["feat"].shape == self.feat.shape and bool((st["feat"].to(self.dev) == self.feat).all())
              and "R" in st and all(p.W.shape == W.shape and p.R.shape == R.shape
                                    for p, W, R in zip(self.parts, st["W"], st["R"])))
        if ok:
            for p, W, R in zip(self.parts, st["W"], st["R"]):
                p.W, p.R = W.to(self.dev), R.to(self.dev)
        return ok


def synthetic_song(n_steps=128, seed=0):
    """Fake song with a density similar to a real one, used while no song is prepared yet."""
    rng = np.random.default_rng(seed)
    parts = {"drums": {"classes": ["kick", "snare", "hat"],
                       "events": [[s, c, 1] for s in range(n_steps) for c, k in enumerate([4, 8, 2]) if s % k == (c == 1) * 4]}}
    for p, density, n_cls in [("bass", .25, 8), ("guitar", .15, 12), ("synth", .2, 16), ("vocals", .2, 12)]:
        parts[p] = {"classes": list(range(48, 48 + n_cls)),
                    "events": [[s, int(rng.integers(n_cls)), 2] for s in range(n_steps) if rng.random() < density]}
    return {"bpm": 126.0, "step_sec": 60 / 126 / 4, "n_steps": n_steps, "parts": parts}


def song_dir(name=None, ask=True):
    """data/songs/<name>; without a name: ask in the terminal (or the first prepared song when ask=False)."""
    import prep
    if name or ask:
        return prep.pick(name)
    have = [d for d in prep.songs() if (d / "song.json").exists()]
    return have[0] if have else None


def load_song(d=None, n_steps=None):
    """The song.json of song directory d (a synthetic song if there is none)."""
    song = json.loads((d / "song.json").read_text()) if d else synthetic_song()
    if n_steps:
        song["n_steps"] = min(song["n_steps"], n_steps)
    return song


def self_check(steps=128, train_passes=3):
    d = song_dir(ask=False)
    print(f"self-check on {d.name if d else 'a synthetic song'}")
    song = load_song(d, steps)
    t = time.time()
    band = Band(song)
    dn = int(band.neurons.super_class.isin(["descending", "motor"]).iloc[band.feat.cpu().numpy()].sum())
    print(f"load + probe {time.time() - t:.1f}s: live circuit {band.N} neurons ({len(band.feat)} outputs, {dn} of them "
          f"descending/motor), antennae={len(band.ear_pos) + len(band.ear_bar)}, dt per sixteenth={band.n_dt}")
    torch.cuda.synchronize()
    t = time.time()
    caches = []
    for _ in range(2):  # 2 simulations of the same recording: one to learn, a fresh one for the test
        band.reset()
        caches.append(torch.stack([band.listen(s) for s in range(steps)]))
    torch.cuda.synchronize()
    ms = (time.time() - t) * 1000 / (2 * steps)
    print(f"(a) {ms:.1f} ms per sixteenth ({ms / band.n_dt * 1000:.0f} us per dt), the song allows "
          f"{song['step_sec'] * 1000:.0f}")
    R = caches[0][:, 0]  # (steps, F) brain 0
    responding = int((R.sum(0) > 0).sum())
    rank = int(torch.linalg.matrix_rank(R - R.mean(0)))
    print(f"(b) active features (outputs x sub-windows): {responding}/{R.shape[1]}, mean spikes per sixteenth "
          f"{R.sum(1).mean():.0f}, position-response rank {rank}/{steps}")
    t = time.time()
    for n in range(train_passes):
        for s in range(steps):
            band.play(caches[0][s], s)
        f1 = [round(p.f1(), 2) for p in band.parts]
        print(f"    rehearsal {n + 1}: F1 {f1}")
    learn_ms = (time.time() - t) * 1000 / (train_passes * steps)
    for s in range(steps):
        band.play(caches[1][s], s, learn=False)
    test = {p.name: round(p.f1(), 2) for p in band.parts}
    print(f"(c) F1 after {train_passes} rehearsals: test (fresh simulation) {test}, learning {learn_ms:.1f} ms per sixteenth")
    return ms, song["step_sec"] * 1000, responding, test


def quality(scores):
    """One number to maximize: right notes first, then right loudness (0-1) and length (sixteenths)."""
    return float(np.mean([v["f1"] - 0.1 * v["vel_mae"] - 0.02 * v["dur_mae"] for v in scores.values()]))


def train(d, minutes=45.0, patience=3, tol=1e-4, p0s=(10.0, 1.0, 0.1)):  # 10 has won every time so far: tried first
    """Offline training on song directory d, no audio, no real time: rehearse from a cached simulation until the
    readout stops improving on a fresh simulation (early stopping), for a few RLS regularizations; keep the best.
    Writes d/readout.pt and d/train-log.json: other songs are untouched."""
    global RLS_P0
    t0 = time.time()
    deadline = t0 + minutes * 60
    song = load_song(d)
    print(f"training on {d.name}: {song['n_steps']} sixteenths, up to {minutes:g} min")
    band = Band(song)
    steps = range(song["n_steps"])
    print(f"simulating the whole song twice ({len(steps)} sixteenths): one to learn from, one fresh to check...")
    caches = []
    for _ in range(2):
        band.reset()
        caches.append(torch.stack([band.listen(s) for s in steps]))
    print(f"  done in {time.time() - t0:.0f}s")
    log, best = [], (-np.inf, None, None)
    for p0 in p0s:
        if time.time() > deadline:
            break
        RLS_P0 = p0
        band.parts = [Part(p, song["parts"][p], song["n_steps"], caches[0].shape[2], band.dev) for p in PARTS]
        run_best, bad, n = -np.inf, 0, 0
        while time.time() < deadline and bad < patience:
            n += 1
            for s in steps:
                band.play(caches[0][s], s)
            train_q = quality({p.name: p.score() for p in band.parts})
            for s in steps:
                band.play(caches[1][s], s, learn=False)
            sc = {p.name: p.score() for p in band.parts}
            q = quality(sc)
            log.append({"p0": p0, "rehearsal": n, "train": round(train_q, 5), "check": round(q, 5),
                        "parts": {k: {m: round(x, 4) for m, x in v.items()} for k, v in sc.items()}})
            print(f"P0={p0:<4} rehearsal {n:2d}: quality {q:.4f} (train {train_q:.4f})  "
                  + "  ".join(f"{k} F1 {v['f1']:.3f} vel±{v['vel_mae']:.3f} len±{v['dur_mae']:.2f}" for k, v in sc.items()))
            if q > run_best + tol:
                run_best, bad = q, 0
                if q > best[0]:
                    best = (q, p0, {"W": [p.W.clone() for p in band.parts], "R": [p.R.clone() for p in band.parts],
                                    "feat": band.feat})
            else:
                bad += 1
        print(f"P0={p0}: best quality {run_best:.4f} ({'early stop' if bad >= patience else 'time up'} after {n} rehearsals)")
    q, p0, state = best
    torch.save(state, d / "readout.pt")
    (d / "train-log.json").write_text(json.dumps({"best_quality": q, "best_p0": p0, "minutes": round((time.time() - t0) / 60, 1),
                                                    "log": log}, indent=1))
    print(f"best: quality {q:.4f} with P0={p0} -> {d / 'readout.pt'} (log in train-log.json), "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["train"]:
        args = sys.argv[2:]
        minutes = next((float(a) for a in args if a.replace(".", "", 1).isdigit()), 45.0)
        name = next((a for a in args if not a.replace(".", "", 1).isdigit()), None)
        train(song_dir(name), minutes)
        sys.exit()
    ms, budget, responding, test = self_check()
    assert ms < budget, "simulation slower than real time"
    assert responding > 100, "too few neurons respond to the antennae"
    assert min(test.values()) > 0.9, "F1 < 0.9: the band isn't learning enough"
    print("OK")
