# Let It Flyppen

Five simulated fruit-fly brains, one per instrument (drums, bass, guitar, synth, vocals), learn to play Tame Impala's *Let It Happen*. You watch them on a 3D stage in the browser, see their neurons fire live, and hear them improve rehearsal after rehearsal.

![The fly band on stage](assets/screenshot.png)

## How it works

| Piece | What it is |
|---|---|
| **Brain** | The FlyWire v783 connectome (139k neurons, 15M connections) run with the leaky integrate-and-fire model of [Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9) on the GPU. Each fly has its own copy. At startup a probe finds the sub-circuit the input can actually drive (~1,700 neurons); the rest stays below threshold and never affects anything, so only that part is simulated. It runs in real time: ~70 ms of compute per sixteenth note, which lasts 122 ms. |
| **Ears** | The 1,055 Johnston's organ neurons of the antenna. Every sixteenth note they receive a spike code for the position in the song (beat within the bar + which bar). Each rehearsal the fly hears the same recording, so it gets the same input spikes. |
| **What learns** | The connectome is **frozen**: synapses never change. Only a linear readout learns, mapping the brain's 616 output neurons (spike counts in 4 time windows per sixteenth) to the notes, plus how loud and how long each note is. It is fitted online with recursive least squares (RLS): rehearsal 1 is chaos, rehearsal 2 is clean. |
| **Notes** | Extracted from the mp3. [Demucs](https://github.com/facebookresearch/demucs) separates the instruments (`htdemucs_ft` for drums/bass/vocals, `htdemucs_6s` for guitar/keys/other). [basic-pitch](https://github.com/spotify/basic-pitch) and pYIN transcribe the pitched parts; for each part the transcription closest to its stem wins (chroma similarity). Drums are split into kick/snare/hi-hat by frequency band. You can also feed a multitrack MIDI file instead. |
| **Sound** | Every instrument plays with samples cut from **its own stem**: Kevin's real voice, the real bass, the real kit. The brains decide which notes, when, how loud and how long. |

Measured on the full song (2,126 sixteenths, 132 bars, 123 bpm), the F1 score (correct notes at the correct sixteenth) per part was:

| Rehearsal | Drums | Bass | Guitar | Synth | Vocals |
|---|---|---|---|---|---|
| 1 | 0.47 | 0.08 | 0.03 | 0.18 | 0.03 |
| 2 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |

How close each transcription is to its stem (`uv run prep.py score`, chroma similarity 0–1): synth 0.80, bass 0.81, vocals 0.53, guitar 0.19 (the guitar is silent for most of the song).

## Setup

You need [uv](https://docs.astral.sh/uv/), an NVIDIA GPU, and your own copy of the song as an mp3 (it is not included).

```bash
uv sync --extra prep                          # Python 3.10 + CUDA torch; "prep" is only needed to prepare the data
uv run prep.py fetch                          # download the connectome and annotations (~135 MB) into data/
uv run prep.py song data/let-it-happen.mp3    # mp3 -> stems -> notes + instrument samples (data/song.json, data/samples)
uv run prep.py fly                            # download flybody and build the fly model (data/fly.glb)
uv run brain.py                               # self-check: real time, active neurons, learning on 8 bars
```

Other prep commands: `check` (stems complete), `score` (transcription vs stems), `samples` (re-cut the samples).

Python is pinned to 3.10 because basic-pitch on Windows only works there (via ONNX); on newer versions it requires a TensorFlow build that doesn't exist.

## Play

Hear the whole song played clean as fast as possible:

```bash
uv run server.py --fresh --bars 0 132 --fast 2
```

The server listens to the whole song once (~3 minutes), runs 2 quick rehearsals in a few seconds, then plays live, already clean. Open <http://localhost:8000> and click **Enter the venue** to turn on the audio (add `#demo` to the URL for visuals only).

To hear the band go from chaos to clean instead:

```bash
uv run server.py --fresh                      # first 8 bars, from scratch, live: 1 messy rehearsal, then clean
```

What the band learns is saved in `data/readout.pt`. Without `--fresh` it picks up where it left off. After regenerating `song.json`, start with `--fresh`.

Or train offline first, with no audio and no real time, then just listen:

```bash
uv run brain.py train 45                      # whole song; early stopping or 45 min; best readout -> data/readout.pt
uv run server.py --bars 0 132                 # plays clean from the first rehearsal
```

Training simulates the song once, rehearses from that cache, checks every rehearsal on a fresh simulation (F1, then velocity and length error), tries a few RLS regularizations and keeps the best (log in `data/train-log.json`).

| Option | Effect |
|---|---|
| `--bars A B` | rehearse bars A to B (default `0 8`) |
| `--fast N` | N quick rehearsals before playing live |
| `--fresh` | forget what was learned and start from scratch |
| `--port P` | serve on another port (default 8000) |

## The stage

![Kevin](assets/kevin.png)

- **Listen** to *Flies* (what the brains play), *Score* (the transcribed notes on the same samples) or *Original* (the real recording, kept in sync with the band).
- **Inside the fly brain**: the selected fly's brain as a 3D point cloud (FlyWire neuron positions); the simulated neurons light up with their real spikes every sixteenth, by firing rate or by cell class. Below it: a raster of 48 output neurons, spike and active-neuron counters, and the readout's top 5 notes.
- **Member cards** show live F1, precision and recall, best and last rehearsal, and mute/solo buttons. Click a card to move the camera to that fly.
- The **piano roll** shows the selected member's expected notes (outlined) against the played ones (green if right, red if wrong).
- The **chart** plots the band's average F1 per rehearsal.
- The flies are the [flybody](https://github.com/TuragaLab/flybody) model, posed as musicians. The vocalist is modelled on Kevin Parker: dark shaggy hair with a fringe, ivory sunglasses with dark tinted lenses, a fluffy white mohair sweater with charcoal Nordic motifs, layered wooden-bead necklaces. Without `data/fly.glb` the stage falls back to simpler procedural flies.
- **Shortcuts**: `1`–`5` select a member, `O` cycles flies/score/original, `H` hides the UI, `I` opens the info panel, `Esc` returns to the stage view.

"Clean" means the flies play exactly the transcribed notes. The timbres are the song's own, but the parts are rebuilt note by note from an automatic transcription, so it is a reconstruction, not the record. A well-made MIDI file gives cleaner targets: `uv run prep.py song file.mid` (then the stage uses synths, since there are no stems to sample).

## Files

| File | Purpose |
|---|---|
| `prep.py` | data: connectome download, stem separation, transcription, samples, flybody model |
| `brain.py` | the five LIF brains on the GPU and the RLS readout |
| `server.py` | endless rehearsals + WebSocket + recording, samples, brain map and fly model for the stage |
| `web/` | 3D stage (Three.js, `stage.js`), brain map (`brainmap.js`), audio and UI (Tone.js, `app.js`) |
| `assets/` | screenshots |

Generated data (`data/`: connectome, stems, samples, fly model) is not in the repository.

## Credits

- Connectome: [FlyWire](https://flywire.ai) v783; model: Shiu et al., *Nature* 2024 ([code](https://github.com/philshiu/Drosophila_brain_model)).
- Fly model: [flybody](https://github.com/TuragaLab/flybody) by Google DeepMind and HHMI Janelia, Apache-2.0 (Vaxenburg et al., *Nature* 2025). `prep.py fly` downloads it and saves its license as `data/flybody-LICENSE`.
