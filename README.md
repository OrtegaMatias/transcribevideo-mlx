<h1 align="center">transcribevideo-mlx</h1>

<p align="center">
  Turn a <b>screen recording</b> into a written report — <b>100% locally</b> on Apple Silicon.<br>
  It reads what's on screen and listens to what's being said, then cross-references both.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-30D158" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10%2B-5AC8FA" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-111?logo=apple" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/runs-100%25%20local-5E5CE6" alt="100% local">
</p>

<p align="center">▚▚▖▘▝▗▚▘▖▝▚▗▘▚▖▝▘▗▚▖▘</p>

Drop a screencast on your terminal and get back a Markdown report: what system is
being shown, the steps taken, the data actually visible on screen, and where the
narration and the interface disagree. No cloud, no API keys, no uploads.

It's the video counterpart to
[transcribe-mlx](https://github.com/OrtegaMatias/transcribe-mlx).

## Why not just transcribe the audio?

Because in a demo or a training recording, half the information is never spoken.
Identifiers, field labels, values, states, error messages — they're on screen and
the narrator says "here you see the profile". A transcript alone loses all of it.

## How it works

```
video ──┬─► sample at 2fps ─► dhash 1024-bit ─► screen cuts ─► representative frames ─┐
        │                                                                             ├─► per-screen analysis ─► report
        └─► MLX Whisper ─► timestamped speech ─► windows snapped to sentence ends ────┘        (VLM)             (text)
```

1. **Screen segmentation.** The video is sampled in grayscale at low resolution and
   each frame hashed. A screen change is a hash distance above threshold. One pass
   gives both the cuts and the dedup — a 30-minute recording collapses to a few
   dozen unique screens instead of hundreds of near-identical frames.

2. **Audio, snapped to speech.** Whisper transcribes with timestamps. Each visual
   cut is then nudged to the nearest sentence boundary, so a window never splits
   mid-sentence. Each screen's audio window is exactly the time that screen was up.

3. **Per-screen analysis.** A local vision model receives the frame *and* the
   narration from that window, and returns structured JSON: text literally read on
   screen, UI elements, what was said, and a synthesis of both.

4. **Continuity repair.** The model also judges whether its window stands on its
   own. If an explanation was cut in half by a screen change, the harness re-runs
   the analysis with both screens merged, so no idea is left dangling.

5. **Report.** A final text-only pass turns all the units into Markdown.

## What you get

Two files in `~/Downloads`:

- **`<name>.md`** — the report: summary, step-by-step walkthrough with timestamps,
  data read from the screen, and observations. Below it, an auditable timeline with
  every screen's extracted text.
- **`<name>.json`** — everything intermediate (frames, timings, per-screen chunks,
  raw transcript), so you can regenerate or post-process without touching the video again.

Plus `<name>-frames/` with the frames that were actually analyzed.

## Requirements

- **macOS on Apple Silicon.** MLX is Apple-Silicon only.
- **Python 3.10+**
- **[ffmpeg](https://ffmpeg.org/)** — `brew install ffmpeg`
- **~20 GB of free RAM** while running, for a 27B 4-bit vision model.

## Installation

```bash
uv tool install git+https://github.com/OrtegaMatias/transcribevideo-mlx
# or: pipx install git+https://github.com/OrtegaMatias/transcribevideo-mlx
```

The first run downloads the vision model (~16 GB, cached afterwards). If you already
have it through LM Studio, it's picked up from `~/.lmstudio/models` instead of being
downloaded again.

## Usage

```bash
transcribevideo                          # prompts you to drag a video in
transcribevideo ~/Downloads/demo.mp4     # or pass a path
transcribevideo a.mp4 b.mov              # several at once
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--vlm` | Vision model (HF repo id or local path) | `lmstudio-community/Qwen3.8-27B-MLX-4bit` |
| `--whisper` | `large-v3-turbo`, `large-v3`, `medium`, `small`, `base`, `tiny` | `large-v3-turbo` |
| `--lang` | Force a language code (`es`, `en`, …) | auto-detect |
| `--fps` | Sampling rate for screen-change detection | `2.0` |
| `--threshold` | dhash distance (of 1024) to call it a new screen | `50` |
| `--min-screen` | Seconds a screen must persist to be analyzed; briefer stretches are folded into the previous one | `2.0` |
| `--max-screens` | Cap on screens analyzed (0 = no cap) | `0` |

### While it runs

The terminal shows the model working rather than a spinner: a progress bar over
screens, **the JSON field being written and its content arriving live** — you watch
the text come off the image, line by line — plus a sparkline of tokens/second,
cumulative prompt and generation tokens, and peak memory. Merges announce
themselves when a truncated idea pulls in the next screen.

<p align="center">▚▚▖▘▝▗▚▘▖▝▚▗▘▚▖▝▘▗▚▖▘</p>

## Design notes

Three findings that shaped this tool. All of them were measured, not assumed — and
all three contradict the obvious approach, so they may save you some time.

### ffmpeg's `scene` filter doesn't work on screencasts

The usual way to find scene changes is `select='gt(scene,0.3)'`. On screen
recordings it finds nothing. That filter measures global pixel difference and is
calibrated for film cuts; two screens of the same application share background,
header and layout, so a real screen change scores around **0.05** — an order of
magnitude below any recommended threshold, and tangled with encoder-keyframe noise.

A 1024-bit dhash separates the same cases by a factor of ~23 (92–148 bits between
different screens, 4 bits for the same screen re-encoded), with a stable threshold
plateau from 20 to 60. Note that the classic 64-bit 8×8 dhash is *not* enough here:
the two most similar screens land 5 bits apart, indistinguishable from noise. UI
needs the finer grid.

### A screen is defined by how long it stays, not by how different it looks

Pixel distance alone over-segments badly. On a real phone screen recording
(5 minutes, animated content), a 1024-bit dhash found **154 distinct screens** —
about 38 minutes of model time for 5 minutes of video. The distribution explains
why: the **median screen lasted 0.5 s**, and 71% lasted under 2 s. Those are
animation frames, not screens. Nobody reads or narrates something that was up for
half a second.

So the second filter is temporal, not visual: a stretch shorter than `--min-screen`
is **folded into the preceding screen** rather than dropped, so the narration
spoken over a transition still belongs somewhere and no time goes unaccounted for.
Of a burst of rapid changes the surviving cut is the *last* one — the moment the
screen settled — so the captured frame is settled content rather than a
half-finished transition. At the 2 s default that recording goes from 154 screens
to 45.

### Local OpenAI-compatible servers may not count image tokens

Running the same image through LM Studio's API reported `prompt_tokens: 75`.
In-process through `mlx-vlm`, the same prompt is **914** tokens. The image is simply
not counted in the reported usage. Since the server returns HTTP 400 when the
context is exceeded rather than truncating, budgeting against the reported number
walks you into a hard failure with no warning. Running the model in-process makes
the number real — which is the main reason this tool loads MLX directly instead of
talking to a local server.

### Turn reasoning off for extraction, on for synthesis

Qwen3.8's chat template defaults to `reasoning_effort: xhigh`. Reading text off a
screenshot is not a reasoning task, and the thinking tokens dominate: the same OCR
call takes 8.3 s with default reasoning and **3.2 s** with `enable_thinking=False`.
Across the dozens of per-screen calls that's most of the runtime. The final report,
which genuinely is synthesis, keeps reasoning on.

## Limitations

- **Screen recordings are the target.** Camera footage, talking heads and general
  video will technically run, but the prompts assume an interface is being shown.
- **OCR contamination is mitigated, not eliminated.** The model gets the narration
  alongside the frame, so it can in principle "read" something it only heard. The
  prompt asks for the literal text before any interpreted field and forbids
  completing from audio, but the honest answer is that this is the failure mode to
  watch for. Compare `texto_en_pantalla` against the frames in the JSON if it matters.
- **Continuity merging is capped at 3 screens.** Beyond that the unit is flagged and
  the final report stitches the idea instead.
- **Runtime scales with unique screens, not video length.** Each screen costs roughly
  10–20 s on an M-series Pro. A calm, static UI walkthrough yields few screens and
  runs fast; a recording full of animation yields many. If a run looks too long, raise
  `--min-screen` before anything else — it's the knob with the most leverage.
- **Whisper timestamps can run past the end of the media** (measured: a segment ending
  at 318.8 s on a 305.9 s video). The video duration bounds them here, but be aware of
  it if you consume the raw transcript from the JSON.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
  and [mlx-whisper](https://github.com/ml-explore/mlx-examples)
- [Qwen](https://github.com/QwenLM) for the vision-language model
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [Rich](https://github.com/Textualize/rich) and [pyfiglet](https://github.com/pwall2222/pyfiglet)

## License

[MIT](LICENSE) © Matias Ortega Carrasco
