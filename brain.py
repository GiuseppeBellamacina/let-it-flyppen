"""Drosophila brains (FlyWire v783 connectome, LIF from Shiu et al. 2024) on GPU, one per instrument.

The connectome is frozen (reservoir): each fly "hears" the position in the song through its
Johnston's organ auditory neurons, and learns only a linear readout from its output neurons
to the notes of its own part.

  uv run brain.py      # self-check: speed, neuron response, learning over 8 bars
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
RLS_P0 = 1.0  # RLS: initial covariance (= 1/regularization); lower = learns slower but smoother
N_RASTER = 48  # output neurons (the most active) shown in the stage raster
EARS = ["auditory", "wind_gravity", "grooming"]  # antennal JO neurons: the "sound" input
PROBE_STEPS = 16
BINS = 4  # spike counts in 4 sub-windows per sixteenth: 4x features (rank) at zero simulation cost


def load_connectome():
    """Return (post, pre, weight_mV) as numpy arrays, plus per-neuron annotations (index = Presynaptic_Index)."""
    ids = pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index
    c = pd.read_parquet(DATA / "Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    ann = (pd.read_csv(DATA / "annotations.tsv", sep="\t", usecols=["root_id", "super_class", "cell_sub_class"],
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
    """A fly's readout: features -> note onsets of its part.

    Linear regression learned online with recursive least squares (RLS): at every sixteenth, the exact
    least-squares solution over everything heard so far. Converges in 1-2 rehearsals even on the whole
    song (NLMS gradient descent needed ~30)."""

    def __init__(self, name, spec, n_steps, n_feat, dev):
        self.name, self.classes, self.mono = name, spec["classes"], name in MONO
        C = len(self.classes)
        Y = torch.zeros(n_steps, C + self.mono)
        for s, c, _ in spec["events"]:
            if s < n_steps:
                Y[s, c] = 1
        if self.mono:  # extra "no onset" class: the note played is the argmax
            Y[:, C] = Y[:, :C].sum(1) == 0
        self.Y = Y.to(dev)
        self.W = torch.zeros(n_feat + 1, Y.shape[1], device=dev)
        self.reset_learning()
        durs = {}
        for _, c, d in spec["events"]:
            durs.setdefault(c, []).append(d)
        self.dur = [int(np.median(durs.get(c, [1]))) for c in range(C)]
        self.tp = self.fp = self.fn = 0

    def reset_learning(self):
        # ponytail: P (~50 MB in float64) is not saved; on resume it restarts from RLS_P0 with the already learned W
        n = self.W.shape[0]
        self.P = torch.eye(n, dtype=torch.float64, device=self.W.device) * RLS_P0

    def __call__(self, x, s, learn=True):
        """x: spike counts (F+1,) with bias. Return (played indices, expected indices)."""
        C, y = len(self.classes), self.Y[s]
        z = x @ self.W
        if self.mono:
            k = int(z.argmax())
            played = [] if k == C else [k]
        else:
            played = (z > 0.5).nonzero().flatten().tolist()
        if learn:
            xd = x.double()
            Px = self.P @ xd
            gain = Px / (1 + xd @ Px)
            self.P -= torch.outer(gain, Px)
            self.W += torch.outer(gain.float(), y - z)
        target = y[:C].nonzero().flatten().tolist()
        hit = len(set(played) & set(target))
        self.tp, self.fp, self.fn = self.tp + hit, self.fp + len(played) - hit, self.fn + len(target) - hit
        return played, target

    def score(self):
        """F1, precision and recall on onsets since the last call (and reset the counters)."""
        tp, fp, fn = self.tp, self.fp, self.fn
        self.tp = self.fp = self.fn = 0
        return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}

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
        torch.manual_seed(0)  # same circuit on every start, so data/readout.pt stays valid
        all_ears = torch.tensor(ears, device=dev)
        fired = sum(self.run(all_ears).sum((0, 2)) for _ in range(PROBE_STEPS)).cpu().numpy()
        alive = np.union1d(np.flatnonzero(fired > 0), ears)
        self.M, self.N = matrix(edges, self.N, alive, dev), len(alive)
        self.neurons = ann.iloc[alive].reset_index(drop=True)  # root_id / class of each simulated neuron
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
        return self.run(self.code(s), seed=s)[:, self.feat].permute(2, 0, 1).reshape(self.B, -1)

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
        """r: (B, BINS*F) from listen. Each fly plays (and learns) its part. Return {part: (played, expected)}."""
        x = torch.cat([r, torch.ones(self.B, 1, device=self.dev)], 1)
        return {p.name: p(x[i], s, learn) for i, p in enumerate(self.parts)}

    def spikes(self, r):
        """Spike counts (B, N_RASTER) of the neurons shown in the raster, from a listen output."""
        return r.view(self.B, BINS, -1).sum(1)[:, self.raster]

    def state(self):
        return {"W": [p.W for p in self.parts], "feat": self.feat}

    def load(self, st):
        """Resume a saved readout only if circuit and classes match. Return whether it was applied."""
        ok = (st["feat"].shape == self.feat.shape and bool((st["feat"].to(self.dev) == self.feat).all())
              and all(p.W.shape == W.shape for p, W in zip(self.parts, st["W"])))
        if ok:
            for p, W in zip(self.parts, st["W"]):
                p.W = W.to(self.dev)
        return ok


def synthetic_song(n_steps=128, seed=0):
    """Fake song with a density similar to a real one, used while data/song.json doesn't exist yet."""
    rng = np.random.default_rng(seed)
    parts = {"drums": {"classes": ["kick", "snare", "hat"],
                       "events": [[s, c, 1] for s in range(n_steps) for c, k in enumerate([4, 8, 2]) if s % k == (c == 1) * 4]}}
    for p, density, n_cls in [("bass", .25, 8), ("guitar", .15, 12), ("synth", .2, 16), ("vocals", .2, 12)]:
        parts[p] = {"classes": list(range(48, 48 + n_cls)),
                    "events": [[s, int(rng.integers(n_cls)), 2] for s in range(n_steps) if rng.random() < density]}
    return {"bpm": 126.0, "step_sec": 60 / 126 / 4, "n_steps": n_steps, "parts": parts}


def load_song(n_steps=None):
    path = DATA / "song.json"
    song = json.loads(path.read_text()) if path.exists() else synthetic_song()
    if n_steps:
        song["n_steps"] = min(song["n_steps"], n_steps)
    return song


def self_check(steps=128, train_passes=3):
    song = load_song(steps)
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


if __name__ == "__main__":
    ms, budget, responding, test = self_check()
    assert ms < budget, "simulation slower than real time"
    assert responding > 100, "too few neurons respond to the antennae"
    assert min(test.values()) > 0.9, "F1 < 0.9: the band isn't learning enough"
    print("OK")
