"""The fly band rehearses nonstop; the 3D stage follows it over WebSocket. The stage can switch song.

  uv run server.py                    # the most recently trained song, whole song, live
  uv run server.py --song NAME        # another song from data/songs (a unique prefix is enough)
  uv run server.py --bars 0 8         # only a section (bars [from, to))
  uv run server.py --fast 2           # 2 quick rehearsals from the cache before performing live
  uv run server.py --fresh            # ignore the song's training and learn live from scratch
  uv run server.py --port 8010        # another port
then open http://localhost:8000

A song trained offline (brain.py train) performs with its training untouched; an untrained one learns live and
saves what it learned in its own folder. Songs never share files.
"""
import argparse
import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import brain
import prep

ap = argparse.ArgumentParser()
ap.add_argument("--song", help="song name in data/songs (default: the most recently trained)")
ap.add_argument("--bars", type=int, nargs=2, help="rehearse only bars [from, to)")
ap.add_argument("--fast", type=int, default=0)
ap.add_argument("--fresh", action="store_true", help="ignore the song's training and learn live from scratch")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()


def catalog():
    """The prepared songs, for the stage's picker."""
    out = []
    for d in prep.songs():
        if (d / "song.json").exists():
            s = json.loads((d / "song.json").read_text())
            log = d / "train-log.json"
            out.append({"slug": d.name, "title": s.get("title", d.name), "bpm": s["bpm"], "bars": s["n_steps"] // 16,
                        "trained": log.exists(), "quality": json.loads(log.read_text()).get("best_quality") if log.exists() else None})
    return out


def archive_training(d):
    """A training that no longer matches the song's notes: keep it aside as *.old, out of the way."""
    for f in ("readout.pt", "train-log.json"):
        if (d / f).exists():
            (d / f).replace(d / f.replace(".", ".old.", 1))
    print(f"{d.name}: its training doesn't match the current notes, archived as readout.old.pt; retrain or let it learn live")


def default_song():
    have = [d for d in prep.songs() if (d / "song.json").exists()]
    if not have:
        raise SystemExit("no songs yet: uv run prep.py song path/to/song.mp3")
    trained = [d for d in have if (d / "readout.pt").exists()]
    return max(trained, key=lambda d: (d / "readout.pt").stat().st_mtime) if trained else have[0]


class Show:
    """The song on stage: its brains, its section, its rehearsals. load() swaps it for another."""

    def __init__(self):
        self.d = self.song = self.band = self.hello = None
        self.steps, self.history, self.task, self.loading = range(0), [], None, False
        self.learn = True

    def url(self, rel):  # a file of the current song, as served by the /songs mount
        return f"/songs/{self.d.name}/{rel}" if rel and (self.d / rel).exists() else None

    async def load(self, d, fresh=False):
        if self.task:
            self.task.cancel()
        self.loading = True
        song = brain.load_song(d)
        await set_status(f"loading {song.get('title', d.name)}: the flies are growing new brains…")
        band = await asyncio.to_thread(brain.Band, song)
        resumed = (d / "readout.pt").exists() and not fresh and band.load(torch.load(d / "readout.pt"))
        if (d / "readout.pt").exists() and not resumed and not fresh:  # notes changed since: that training is stale
            archive_training(d)
        trained = resumed and (d / "train-log.json").exists()
        # trained offline: perform with the training untouched; otherwise learn live (and save in the song's folder)
        self.learn = not trained
        a, b = args.bars if args.bars and self.d is None else (0, song["n_steps"] // 16 + 1)
        self.d, self.song, self.band, self.history = d, song, band, []
        self.steps = range(a * 16, min(b * 16, song["n_steps"]))
        print(f"{d.name}: {song['bpm']} bpm, sixteenths {self.steps.start}-{self.steps.stop}, "
              f"{'trained, performing' if not self.learn else 'resumed, learning live' if resumed else 'learning live from scratch'}")
        self.hello = {
            "t": "hello", "song": {"slug": d.name, "title": song.get("title", d.name), "trained": trained,
                                   "learning": self.learn},
            "songs": catalog(), "bpm": song["bpm"], "step_sec": song["step_sec"],
            "steps": [self.steps.start, self.steps.stop], "song_steps": song["n_steps"],
            "neurons": band.N, "outputs": len(band.feat), "ears": len(band.ear_pos) + len(band.ear_bar),
            "features": brain.BINS * len(band.feat), "learner": "RLS",
            "parts": {p.name: {"classes": p.classes, "mono": p.mono} for p in band.parts},
            "audio": self.url(song.get("audio")) if "times" in song else None,  # "Original"
            # per part: the stem its fly plays back in stem mode (prep.py web_audio)
            "stems": {p: u for p, rel in song.get("stems", {}).items() if p != "mix" and (u := self.url(rel))} or None,
            # instrument samples cut from the stems (prep.py samples); the stage falls back to synths without them
            "samples": {part: {c: u for c, rel in files.items() if (u := self.url(rel))}
                        for part, files in song.get("samples", {}).items()} or None,
            "brain": f"/brain.json?song={d.name}", "alive": band.N,
            "fly": "/fly.glb" if (prep.DATA / "fly.glb").exists() else None}
        self.loading = False
        await send({**self.hello, "history": self.history})
        self.task = asyncio.create_task(rehearse(self))


show, clients, status = Show(), set(), {"t": "status", "text": ""}


async def send(msg):
    text = json.dumps(msg)
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)


async def set_status(text):
    status["text"] = text
    await send(status)


def notes(part, played):
    """[(class index, length, velocity)] -> [[class value, length in sixteenths, velocity 0-1]]."""
    return [[part.classes[c], length, vel] for c, length, vel in played]


def b64(array):
    return base64.b64encode(array.tobytes()).decode()


async def end_pass(s, fast):
    sc = {p.name: p.score() for p in s.band.parts}
    f1 = {k: round(v["f1"], 3) for k, v in sc.items()}
    s.history.append(f1)
    if s.learn and not (s.d / "train-log.json").exists():  # learning live: save in the song's own folder
        torch.save(s.band.state(), s.d / "readout.pt")  # (an offline training never reaches here: learn is False)
    print(f"{s.d.name} rehearsal {len(s.history)}{' (quick)' if fast else ''}: " + " ".join(f"{k} {v:.2f}" for k, v in f1.items()))
    await send({"t": "pass", "n": len(s.history), "f1": f1, "fast": fast,
                "precision": {k: round(v["precision"], 3) for k, v in sc.items()},
                "recall": {k: round(v["recall"], 3) for k, v in sc.items()}})


async def rehearse(s):
    band, song, steps = s.band, s.song, s.steps
    times = song.get("times") or [i * song["step_sec"] for i in range(song["n_steps"])]
    if args.fast and s.learn:  # frozen reservoir + same recording: one simulated pass is enough for many quick rehearsals
        await set_status("the flies are listening to the song for the quick rehearsals…")
        band.reset()
        cache = [await asyncio.to_thread(band.listen, i) for i in steps]
        for n in range(args.fast):
            await set_status(f"quick rehearsal {n + 1}/{args.fast}…")
            await asyncio.to_thread(lambda: [band.play(r, i) for r, i in zip(cache, steps)])
            await end_pass(s, True)
    await set_status("")
    while True:  # live: real simulation, paced on the song's own sixteenths (they follow its tempo)
        band.reset()  # every rehearsal starts from the same state: same recording, same brain
        t0, first = time.perf_counter(), times[steps.start]
        await send({"t": "start", "n": len(s.history) + 1})
        for i in steps:
            r = await asyncio.to_thread(band.listen, i)
            out = band.play(r, i, learn=s.learn)
            await send({"t": "step", "i": i, "act": r.sum(1).tolist(), "time": times[i],
                        "raster": dict(zip(brain.PARTS, band.spikes(r).int().tolist())),
                        # spike count of every simulated neuron, per brain: uint8[B * N], brain-major
                        "spikes": b64(band.last_counts.T.clamp(max=255).to(torch.uint8).cpu().numpy()),
                        "scores": {p.name: p.top() for p in band.parts},
                        "play": {p.name: notes(p, out[p.name][0]) for p in band.parts},
                        "target": {p.name: notes(p, out[p.name][1]) for p in band.parts}})
            nxt = times[i + 1] if i + 1 < len(times) else times[i] + song["step_sec"]
            await asyncio.sleep(max(0, t0 + (nxt - first) - time.perf_counter()))
        await end_pass(s, False)


@contextlib.asynccontextmanager
async def lifespan(_):
    await show.load(prep.pick(args.song) if args.song else default_song(), fresh=args.fresh)
    yield
    if show.task:
        show.task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/brain.json")
def brain_map():
    """Neuron positions for the stage map: a brain silhouette and the simulated neurons (in spike order)."""
    b = show.band
    return {"shape": b64(b.shape_pos), "alive": b64(b.alive_pos), "classes": b.alive_class,
            "n_shape": len(b.shape_pos), "n_alive": b.N}


@app.get("/fly.glb")
async def fly():
    """The flybody fly model converted by prep.py fly (absent = the stage keeps its procedural flies)."""
    path = prep.DATA / "fly.glb"
    if not path.exists():
        raise HTTPException(404, "run `uv run prep.py fly` to build the flybody model")
    return FileResponse(path)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    if show.hello:
        await sock.send_text(json.dumps({**show.hello, "history": show.history}))
    if status["text"]:
        await sock.send_text(json.dumps(status))
    clients.add(sock)
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("t") == "song" and not show.loading and msg.get("slug") != show.d.name:
                hit = [d for d in prep.songs() if d.name == msg.get("slug") and (d / "song.json").exists()]
                if hit:
                    asyncio.create_task(show.load(hit[0]))
    except (WebSocketDisconnect, json.JSONDecodeError):
        clients.discard(sock)


# mounted after the routes above: "/" is a catch-all
app.mount("/songs", StaticFiles(directory=prep.SONGS, check_dir=False))  # each song's audio, stems, samples
app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port)
