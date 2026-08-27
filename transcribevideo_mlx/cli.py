#!/usr/bin/env python3
"""transcribevideo — analiza una grabación de pantalla, local, en Apple Silicon.

    transcribevideo                       # te pide arrastrar el video
    transcribevideo ~/Downloads/demo.mp4  # ruta directa

Salida: ~/Downloads/<nombre>.md y <nombre>.json (no sobrescribe).
"""
from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
import time
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

try:
    import pyfiglet
except Exception:  # noqa: BLE001 - el wordmark es decorativo
    pyfiglet = None

from . import __version__
from .audio import MODELS, snap_to_speech
from .report import render_json, render_markdown, unique_path, write_report
from .segment import CUT_THRESHOLD, SAMPLE_FPS
from .vlm import DEFAULT_MODEL

# Mismo tema que transcribe-mlx: cian → índigo, verde de éxito.
C1, C2 = "#5AC8FA", "#5E5CE6"
OK = "#30D158"
ERR = "#FF453A"
DIM = "grey50"
BARS = " ▁▂▃▄▅▆▇█"
SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DOWNLOADS = Path.home() / "Downloads"

console = Console(file=sys.stdout, highlight=False)


# ── color ───────────────────────────────────────────────────────────────────
def interp(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    r = tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))
    return f"#{r[0]:02x}{r[1]:02x}{r[2]:02x}"


def gradient(s: str, c1: str = C1, c2: str = C2) -> Text:
    lines = s.rstrip("\n").split("\n")
    width = max((len(line) for line in lines), default=1) or 1
    out = Text()
    for line in lines:
        for i, char in enumerate(line):
            out.append(char, style=interp(c1, c2, i / (width - 1) if width > 1 else 0.0))
        out.append("\n")
    return out


def human(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ── piezas de UI ────────────────────────────────────────────────────────────
def banner(vlm: str, whisper: str) -> Group:
    motif = "▚▚▖▘▝▗▚▘▖▝▚▗▘▚▖▝▘▗▚▖▘"
    word = pyfiglet.figlet_format("transcribevideo", font="small") if pyfiglet else "transcribevideo"
    sub = Text("local · mlx · visión + audio · apple silicon", style=DIM, justify="center")
    cfg = Text(justify="center")
    cfg.append("visión ", style="grey42"); cfg.append(vlm.split("/")[-1], style="grey70")
    cfg.append("   ·   ", style="grey30")
    cfg.append("audio ", style="grey42"); cfg.append(whisper, style="grey70")
    cfg.append("   ·   ", style="grey30")
    cfg.append("destino ", style="grey42"); cfg.append("~/Downloads", style="grey70")
    return Group(Text(), Align.center(gradient(motif)), Align.center(gradient(word)),
                 sub, Text(), cfg, Text())


def bar(done: int, total: int, width: int) -> Text:
    if total <= 0:
        return Text()
    filled = int(round(width * done / total))
    out = Text()
    for i in range(width):
        style = interp(C1, C2, i / (width - 1) if width > 1 else 0.0)
        out.append("█" if i < filled else "░",
                   style=style if i < filled else "grey27")
    return out


def wave(frame: int, width: int) -> Text:
    out = Text(justify="center")
    for c in range(width):
        v = (math.sin(c * 0.45 - frame * 0.5) + math.sin(c * 0.27 - frame * 0.31)) / 2
        level = int(round(((v + 1) / 2) * (len(BARS) - 1)))
        out.append(BARS[level], style=interp(C1, C2, c / (width - 1) if width > 1 else 0.0))
    return out


class Stage:
    """Panel vivo compartido por todas las etapas de una corrida."""

    def __init__(self, live: Live, name: str, width: int):
        self.live, self.name, self.width = live, name, width
        self.frame = 0
        self.label = ""
        self.done = self.total = 0
        self.recent: list[str] = []
        self.started = time.time()

    def render(self) -> Panel:
        head = Text(justify="center")
        head.append(SPIN[self.frame % len(SPIN)] + "  ", style=C1)
        head.append(self.label or self.name, style="grey74")
        head.append(f"    {human(time.time() - self.started)}", style=DIM)

        parts: list = []
        if self.total:
            inner = max(20, min(self.width - 16, 44))
            counter = Text(f"  {self.done}/{self.total}", style=DIM)
            parts += [Align.center(Group(bar(self.done, self.total, inner), counter)), Text()]
        else:
            parts += [wave(self.frame, max(10, min(self.width - 12, 44))), Text()]
        parts.append(head)

        if self.recent:
            parts.append(Text())
            for line in self.recent[-4:]:
                item = Text("   ")
                item.append("· ", style="grey35")
                item.append(line[:self.width - 22], style="grey58")
                parts.append(item)

        return Panel(Align.center(Group(*parts)), box=box.ROUNDED,
                     border_style="grey35", padding=(1, 4),
                     title=Text(f" {self.name} ", style=DIM), title_align="left")

    def tick(self) -> None:
        self.frame += 1
        self.live.update(self.render())


def result_panel(video: Path, md: Path, js: Path, frames_dir: Path,
                 units, duration: float, elapsed: float, body: str) -> Panel:
    preview = body.strip()
    truncated = len(preview) > 700
    if truncated:
        preview = preview[:700].rsplit(" ", 1)[0] + " …"

    merged = sum(1 for u in units if u.merged)
    failed = sum(1 for u in units if u.error)
    sep = "   ·   "

    meta = Text()
    meta.append("◷  ", style=C1)
    meta.append("video ", style="grey42"); meta.append(human(duration), style="grey74")
    meta.append(sep, style="grey30")
    meta.append("en ", style="grey42"); meta.append(human(elapsed), style="grey74")
    meta.append(sep, style="grey30")
    meta.append(f"{len(units)}", style="grey74"); meta.append(" unidades", style="grey42")
    if merged:
        meta.append(sep, style="grey30")
        meta.append(f"{merged}", style="grey74"); meta.append(" fusionadas", style="grey42")
    if failed:
        meta.append(sep, style="grey30")
        meta.append(f"{failed}", style=ERR); meta.append(" con error", style="grey42")

    saved = Text()
    for path in (md, js):
        saved.append("→ ", style=OK)
        saved.append(f"{path}\n", style="grey66 underline")
    saved.append("→ ", style="grey35")
    saved.append(f"{frames_dir}/", style="grey42")

    return Panel(
        Group(Text(preview, style="grey85"),
              *([Text(), Text("vista previa — informe completo en el archivo",
                              style="grey42 italic")] if truncated else []),
              Rule(style="grey30"), meta, Text(), saved),
        box=box.ROUNDED, border_style=OK, padding=(1, 3),
        title=Text(f" ✓  {video.name} ", style=f"bold {OK}"), title_align="left")


def error_panel(name: str, message: str) -> Panel:
    return Panel(Text(message, style="grey78"), box=box.ROUNDED, border_style=ERR,
                 padding=(1, 3), title=Text(f" ✗  {name} ", style=f"bold {ERR}"),
                 title_align="left")


# ── motor ───────────────────────────────────────────────────────────────────
def process(video: Path, args) -> bool:
    from . import audio as audio_mod
    from . import pipeline as pipeline_mod
    from . import segment as segment_mod
    from .vlm import ModelError, VisionModel

    if not video.exists():
        console.print(error_panel(video.name, f"No existe el archivo:\n{video}"))
        return False

    started = time.time()
    frames_dir = DOWNLOADS / f"{video.stem}-frames"
    width = console.width

    with Live(console=console, refresh_per_second=16, transient=True) as live:
        stage = Stage(live, video.name, width)

        # 1 · segmentar
        stage.label = "detectando pantallas"
        stage.tick()
        duration = segment_mod.probe_duration(video)
        hashes = segment_mod.sample_hashes(video, args.fps)
        cuts = segment_mod.find_cuts(hashes, args.fps, args.threshold)
        if args.max_screens and len(cuts) > args.max_screens:
            keep = max(1, len(cuts) // args.max_screens)
            cuts = cuts[::keep][:args.max_screens]
            stage.recent.append(f"limitado a {len(cuts)} pantallas (--max-screens)")
        stage.recent.append(f"{len(cuts)} pantallas únicas en {human(duration)}")
        stage.tick()

        # 2 · audio
        stage.label = "transcribiendo audio"
        stage.tick()
        transcript = audio_mod.transcribe(video, args.whisper, args.lang)
        cuts = snap_to_speech(cuts, transcript)
        stage.recent.append(
            f"{len(transcript.utterances)} segmentos · idioma {transcript.language}")
        stage.tick()

        screens = segment_mod.extract_screens(video, cuts, duration, frames_dir)

        # 3 · cargar el modelo de visión
        stage.label = f"cargando {args.vlm.split('/')[-1]}"
        stage.tick()
        try:
            model = VisionModel(args.vlm)
        except ModelError as exc:
            live.stop()
            console.print(error_panel(video.name, str(exc)))
            return False

        # 4 · analizar pantalla por pantalla
        stage.label = "analizando pantallas"
        stage.total = len(screens)
        stage.tick()

        def on_call(index: int, span: int) -> None:
            stage.done = index
            stage.label = ("analizando pantallas" if span == 1
                           else f"fusionando {span} pantallas (idea truncada)")
            stage.tick()

        def on_unit(unit) -> None:
            stage.done = min(stage.done + len(unit.screens), stage.total)
            mark = " ⇢" if unit.merged else ""
            stage.recent.append(f"{unit.title}{mark}")
            stage.tick()

        units = pipeline_mod.analyze(model, screens, transcript,
                                     on_call=on_call, on_unit=on_unit)

        # 5 · informe
        stage.label = "redactando el informe"
        stage.total = 0
        stage.tick()
        body = write_report(model, units, transcript)

    elapsed = time.time() - started
    md_path = unique_path(DOWNLOADS, video.stem, "md")
    md_path.write_text(
        render_markdown(video, body, units, transcript, duration, elapsed),
        encoding="utf-8")
    json_path = unique_path(DOWNLOADS, video.stem, "json")
    json_path.write_text(
        render_json(video, units, transcript, duration, elapsed, args.vlm, body),
        encoding="utf-8")

    console.print(result_panel(video, md_path, json_path, frames_dir,
                               units, duration, elapsed, body))
    console.print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analiza una grabación de pantalla cruzando imagen y audio, 100% local.")
    parser.add_argument("--version", action="version",
                        version=f"transcribevideo-mlx {__version__}")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--vlm", default=DEFAULT_MODEL,
                        help="modelo de visión MLX (repo HF o ruta local)")
    parser.add_argument("--whisper", choices=list(MODELS), default="large-v3-turbo",
                        help="modelo Whisper para el audio")
    parser.add_argument("--lang", default=None, help="fuerza el idioma (es, en, …)")
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS,
                        help="frecuencia de muestreo para detectar cambios de pantalla")
    parser.add_argument("--threshold", type=int, default=CUT_THRESHOLD,
                        help="distancia dhash (de 1024) para considerar otra pantalla")
    parser.add_argument("--max-screens", type=int, default=0,
                        help="tope de pantallas a analizar (0 = sin tope)")
    args = parser.parse_args()

    console.print(banner(args.vlm, args.whisper))

    files = args.files
    if not files:
        console.print(Align.center(
            Text("  ▶  arrastra el video aquí y presiona Enter  ", style=f"bold {C1}")))
        console.print()
        try:
            files = shlex.split(console.input("  ➜ "))
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 1
        console.print()

    paths = [Path(os.path.expanduser(f)).resolve() for f in files if f.strip()]
    if not paths:
        console.print(error_panel("entrada", "No se indicó ningún video."))
        return 1

    ok = 0
    for path in paths:
        try:
            ok += process(path, args)
        except KeyboardInterrupt:
            console.print()
            console.print(error_panel(path.name, "Interrumpido."))
            return 1
        except Exception as exc:  # noqa: BLE001 - un video roto no tumba el lote
            console.print(error_panel(path.name, f"{type(exc).__name__}: {exc}"))

    if len(paths) > 1:
        console.print(Align.center(
            Text(f"listo · {ok}/{len(paths)} procesado(s)", style=DIM)))
    return 0 if ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
