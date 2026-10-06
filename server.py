"""The fly band rehearses nonstop; the 3D stage follows it over WebSocket.

  uv run server.py                    # rehearse the first 8 bars, live
  uv run server.py --bars 0 132       # another section (bars [from, to)), here the whole song
  uv run server.py --fast 2           # 2 quick rehearsals from the cache before performing live
  uv run server.py --fresh            # forget what was learned (data/readout.pt) and start from scratch
then open http://localhost:8000
"""
import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import brain

READOUT = brain.DATA / "readout.pt"
ap = argparse.ArgumentParser()
ap.add_argument("--bars", type=int, nargs=2, default=[0, 8])
ap.add_argument("--fast", type=int, default=0)
ap.add_argument("--fresh", action="store_true", help="ignore data/readout.pt and start from scratch")
args = ap.parse_args()

song = brain.load_song()
steps = range(args.bars[0] * 16, min(args.bars[1] * 16, song["n_steps"]))
print(f"{song['bpm']} bpm, rehearsing sixteenths {steps.start}-{steps.stop}: loading the connectome...")
band = brain.Band(song)
if READOUT.exists() and not args.fresh:
    print("readout resumed from", READOUT) if band.load(torch.load(READOUT)) else print("saved readout incompatible: starting from scratch")
clients, history, status = set(), [], {"t": "status", "text": ""}
hello = {"t": "hello", "bpm": song["bpm"], "step_sec": song["step_sec"], "steps": [steps.start, steps.stop],
         "neurons": band.N, "outputs": len(band.feat), "ears": len(band.ear_pos) + len(band.ear_bar),
         "features": brain.BINS * len(band.feat), "learner": "RLS", "song_steps": song["n_steps"],
         "audio": "/audio" if "audio" in song and "times" in song and (brain.DATA / song["audio"]).exists() else None,
         "parts": {p.name: {"classes": p.classes, "mono": p.mono} for p in band.parts}}


async def send(msg):
    text = json.dumps(msg)
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)


def notes(part, idx):
    return [[part.classes[i], part.dur[i]] for i in idx]


async def end_pass(fast):
    sc = {p.name: p.score() for p in band.parts}
    f1 = {k: round(v["f1"], 3) for k, v in sc.items()}
    history.append(f1)
    torch.save(band.state(), READOUT)
    print(f"rehearsal {len(history)}{' (quick)' if fast else ''}: " + " ".join(f"{k} {v:.2f}" for k, v in f1.items()))
    await send({"t": "pass", "n": len(history), "f1": f1, "fast": fast,
                "precision": {k: round(v["precision"], 3) for k, v in sc.items()},
                "recall": {k: round(v["recall"], 3) for k, v in sc.items()}})


async def rehearse():
    if args.fast:  # frozen reservoir + same recording: one simulated pass is enough for many quick rehearsals
        status["text"] = "the flies are listening to the song for the quick rehearsals…"
        await send(status)
        band.reset()
        cache = [await asyncio.to_thread(band.listen, s) for s in steps]
        for n in range(args.fast):
            status["text"] = f"quick rehearsal {n + 1}/{args.fast}…"
            await send(status)
            await asyncio.to_thread(lambda: [band.play(r, s) for r, s in zip(cache, steps)])
            await end_pass(True)
    status["text"] = ""
    await send(status)
    while True:  # live: real simulation, in time with the music
        band.reset()  # every rehearsal starts from the same state: same recording, same brain
        t0 = time.perf_counter()
        await send({"t": "start", "n": len(history) + 1})
        for j, s in enumerate(steps):
            r = await asyncio.to_thread(band.listen, s)
            out = band.play(r, s)
            await send({"t": "step", "i": s, "act": r.sum(1).tolist(), "time": song["times"][s] if "times" in song else None,
                        "raster": dict(zip(brain.PARTS, band.spikes(r).int().tolist())),
                        "play": {p.name: notes(p, out[p.name][0]) for p in band.parts},
                        "target": {p.name: notes(p, out[p.name][1]) for p in band.parts}})
            await asyncio.sleep(max(0, t0 + (j + 1) * song["step_sec"] - time.perf_counter()))
        await end_pass(False)


@contextlib.asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(rehearse())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/audio")
async def audio():
    """The original recording, for the stage's "Original" button."""
    return FileResponse(brain.DATA / song["audio"])


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    await sock.send_text(json.dumps({**hello, "history": history}))
    if status["text"]:
        await sock.send_text(json.dumps(status))
    clients.add(sock)
    try:
        while True:
            await sock.receive_text()
    except WebSocketDisconnect:
        clients.discard(sock)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
