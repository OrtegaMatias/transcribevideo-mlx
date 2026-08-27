#!/usr/bin/env python3
"""transcribevideo — analiza una grabación de pantalla, local, en Apple Silicon.

    transcribevideo                       # te pide arrastrar el video
    transcribevideo ~/Downloads/demo.mp4  # ruta directa

Salida: ~/Downloads/<nombre>.md y <nombre>.json (no sobrescribe).
"""
from __future__ import annotations

import argparse
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
from .live import (C1, C2, DIM, ERR, OK, WARN, RunState, ScreenRow, clock,
                   compact, gradient, human, render)
from .report import render_json, render_markdown, unique_path, write_report
from .segment import CUT_THRESHOLD, MIN_SCREEN_SECONDS, SAMPLE_FPS
from .vlm import DEFAULT_MODEL, DEFAULT_REPORTER, Usage, resolve_model_path

DOWNLOADS = Path.home() / "Downloads"
#: Mínimo entre repintados. El modelo emite decenas de tokens por segundo y
#: redibujar en cada uno gasta más CPU en la UI que en la inferencia.
REDRAW_INTERVAL = 0.08

console = Console(file=sys.stdout, highlight=False)


def banner(vlm: str, reporter: str, whisper: str) -> Group:
    motif = "▚▚▖▘▝▗▚▘▖▝▚▗▘▚▖▝▘▗▚▖▘"
    word = pyfiglet.figlet_format("transcribevideo", font="small") if pyfiglet else "transcribevideo"
    sub = Text("local · mlx · visión + audio · apple silicon", style=DIM, justify="center")

    def short(name: str) -> str:
        return name.split("/")[-1].replace("-it-QAT", "").replace("-MLX-4bit", "")

    cfg = Text(justify="center")
    cfg.append("lee ", style="grey42"); cfg.append(short(vlm), style="grey70")
    cfg.append("   ·   ", style="grey30")
    cfg.append("redacta ", style="grey42"); cfg.append(short(reporter), style="grey70")
    cfg.append("   ·   ", style="grey30")
    cfg.append("oye ", style="grey42"); cfg.append(whisper, style="grey70")
    return Group(Text(), Align.center(gradient(motif)), Align.center(gradient(word)),
                 sub, Text(), cfg, Text())


class View:
    """Repintado con throttling sobre la vista viva.

    El modelo emite decenas de tokens por segundo; redibujar en cada uno
    gastaría más CPU en la interfaz que en la inferencia. Se repinta a ritmo
    fijo y el `frame` avanza igual, que es lo que anima el spinner, el latido de
    la fila activa y el cursor.
    """

    def __init__(self, live: Live, state: RunState, width: int, height: int):
        self.live, self.state = live, state
        self.width, self.height = width, height
        self._last = 0.0

    def stage(self, key: str, label: str, detail: str = "") -> None:
        self.state.stage_key = key
        self.state.stage = label
        self.state.detail = detail
        self.draw(force=True)

    def draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < REDRAW_INTERVAL:
            return
        self._last = now
        self.state.frame += 1
        self.live.update(render(self.state, self.width, self.height))


def _short_model(name: str) -> str:
    return (name.split("/")[-1].replace("-it-QAT", "").replace("-MLX-4bit", "")
            .replace("-GGUF", ""))


def model_size_gb(model: str) -> float:
    path = Path(resolve_model_path(model))
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9


def result_panel(video: Path, md: Path, js: Path, frames_dir: Path,
                 units, duration: float, elapsed: float, body: str,
                 usage: Usage) -> Panel:
    preview = body.strip()
    truncated = len(preview) > 700
    if truncated:
        preview = preview[:700].rsplit(" ", 1)[0] + " …"

    screens = sum(len(u.screens) for u in units)
    merged = sum(1 for u in units if u.merged)
    failed = sum(1 for u in units if u.error)
    sep = "   ·   "

    work = Text()
    work.append("◷  ", style=C1)
    work.append("video ", style="grey42"); work.append(clock(duration), style="grey74")
    work.append(sep, style="grey30")
    work.append("en ", style="grey42"); work.append(human(elapsed), style="grey74")
    work.append(sep, style="grey30")
    work.append(f"{screens}", style="grey74"); work.append(" pantallas", style="grey42")
    work.append(" en ", style="grey42")
    work.append(f"{len(units)}", style="grey74"); work.append(" unidades", style="grey42")
    if merged:
        work.append(sep, style="grey30")
        work.append(f"{merged}", style=WARN); work.append(" fusionadas", style="grey42")
    if failed:
        work.append(sep, style="grey30")
        work.append(f"{failed}", style=ERR); work.append(" con error", style="grey42")

    tokens = Text()
    tokens.append("⚡  ", style=C2)
    tokens.append(f"{usage.calls}", style="grey74")
    tokens.append(" llamadas", style="grey42")
    tokens.append(sep, style="grey30")
    tokens.append(compact(usage.prompt_tokens), style="grey74")
    tokens.append(" tokens in", style="grey42")
    tokens.append(sep, style="grey30")
    tokens.append(compact(usage.generation_tokens), style="grey74")
    tokens.append(" out", style="grey42")
    if usage.peak_memory:
        tokens.append(sep, style="grey30")
        tokens.append(f"{usage.peak_memory:.1f} GB", style="grey74")
        tokens.append(" pico", style="grey42")

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
              Rule(style="grey30"), work, tokens, Text(), saved),
        box=box.ROUNDED, border_style=OK, padding=(1, 3),
        title=Text(f" ✓  {video.name} ", style=f"bold {OK}"), title_align="left")


def error_panel(name: str, message: str) -> Panel:
    return Panel(Text(message, style="grey78"), box=box.ROUNDED, border_style=ERR,
                 padding=(1, 3), title=Text(f" ✗  {name} ", style=f"bold {ERR}"),
                 title_align="left")


def process(video: Path, args) -> bool:
    from . import audio as audio_mod
    from . import pipeline as pipeline_mod
    from . import segment as segment_mod
    from .vlm import ModelError, VisionModel

    if not video.exists():
        console.print(error_panel(video.name, f"No existe el archivo:\n{video}"))
        return False

    state = RunState(name=video.name,
                     reader_model=_short_model(args.vlm),
                     writer_model=_short_model(args.reporter))
    frames_dir = DOWNLOADS / f"{video.stem}-frames"
    total_usage = Usage()

    # Pantalla completa cuando hay terminal: la vista ocupa todo el alto y al
    # terminar se restaura lo que había detrás. Sin terminal (salida
    # redirigida) se degrada para no ensuciar el archivo.
    full_screen = console.is_terminal
    height = console.size.height if full_screen else 30

    with Live(console=console, refresh_per_second=20, screen=full_screen,
              transient=not full_screen) as live:
        view = View(live, state, console.width, height)

        # 1 · segmentar
        view.stage("segmentar", "detectando cambios de pantalla")
        duration = segment_mod.probe_duration(video)
        state.detail = f"{clock(duration)} de video"
        view.draw(force=True)

        hashes = segment_mod.sample_hashes(video, args.fps)
        raw_cuts = segment_mod.find_cuts(hashes, args.fps, args.threshold)
        cuts = segment_mod.absorb_transients(raw_cuts, duration, args.min_screen)
        state.detail = f"{len(cuts)} pantallas únicas de {len(hashes)} muestras"
        if len(raw_cuts) > len(cuts):
            state.notes.append(
                f"{len(raw_cuts) - len(cuts)} tramos de menos de {args.min_screen:g}s "
                "fundidos en la pantalla anterior (animaciones)")
        view.draw(force=True)

        # 2 · audio
        state.active_model = f"whisper {args.whisper}"
        view.stage("oír", "transcribiendo el audio", f"whisper {args.whisper}")
        transcript = audio_mod.transcribe(video, args.whisper, args.lang)
        if transcript.utterances:
            state.detail = (f"{len(transcript.utterances)} segmentos · "
                            f"idioma {transcript.language}")
        else:
            state.detail = "sin pista de audio — solo visión"
            state.notes.append("el video no trae audio; el informe sale solo de las pantallas")
        view.draw(force=True)

        cuts = snap_to_speech(cuts, transcript, duration=duration)

        # 3 · extraer los frames elegidos
        state.active_model = ""
        view.stage("extraer", "extrayendo frames")
        screens = segment_mod.extract_screens(
            video, cuts, duration, frames_dir,
            on_skip=lambda at, why: state.notes.append(
                f"pantalla en {clock(at)} descartada, no se pudo extraer"))
        if not screens:
            live.stop()
            console.print(error_panel(video.name, "No se pudo extraer ninguna pantalla."))
            return False

        if args.max_screens and len(screens) > args.max_screens:
            step = max(1, len(screens) // args.max_screens)
            dropped = len(screens) - len(screens[::step][:args.max_screens])
            screens = screens[::step][:args.max_screens]
            state.notes.append(f"{dropped} pantallas omitidas por --max-screens")

        # 4 · cargar el modelo lector
        size = model_size_gb(args.vlm)
        state.active_model = state.reader_model
        view.stage("leer", "cargando el modelo lector",
                   state.reader_model + (f" · {size:.1f} GB" if size else ""))
        try:
            model = VisionModel(args.vlm)
        except ModelError as exc:
            live.stop()
            console.print(error_panel(video.name, str(exc)))
            return False

        # 5 · analizar pantalla por pantalla
        state.screens_total = len(screens)
        state.rows = [ScreenRow(index=i, at=s.start) for i, s in enumerate(screens)]
        view.stage("leer", "leyendo las pantallas", f"{len(screens)} pantallas")

        def on_call(index: int, span: int) -> None:
            state.screens_done = index
            state.active = index
            state.merging = span if span > 1 else 0
            for row in state.rows[index:index + span]:
                row.status = "active"
            first = screens[index]
            last = screens[min(index + span - 1, len(screens) - 1)]
            state.current_label = (f"pantalla {index + 1} de {len(screens)}   ·   "
                                   f"{clock(first.start)} – {clock(last.end)}")
            state.reader.reset()
            view.draw(force=True)

        def on_delta(delta: str) -> None:
            state.reader.feed(delta)
            # El título aparece primero en el JSON: en cuanto se lee, la fila
            # activa deja de decir "leyendo…" y muestra de qué pantalla se trata.
            if state.reader.field == "titulo":
                lines = state.reader.lines(limit=1, width=60)
                if lines:
                    state.rows[state.active].title = lines[0]
            view.draw()

        def on_unit(unit) -> None:
            state.record_unit(unit.usage.seconds)
            start = state.active
            for offset, _screen in enumerate(unit.screens):
                row = state.rows[start + offset]
                row.status = ("error" if unit.error
                              else "merged" if unit.merged else "done")
                row.title = unit.title if offset == 0 else "…continúa"
            state.screens_done = min(start + len(unit.screens), state.screens_total)
            state.merging = 0
            total_usage.add(unit.usage)
            state.prompt_tokens = total_usage.prompt_tokens
            state.generation_tokens = total_usage.generation_tokens
            state.peak_memory = total_usage.peak_memory
            state.record_tps(unit.usage.tps)
            if unit.error:
                state.notes.append(f"{clock(unit.start)} sin analizar: {unit.error[:60]}")
            state.reader.reset()
            view.draw(force=True)

        units = pipeline_mod.analyze(model, screens, transcript,
                                     on_call=on_call, on_unit=on_unit,
                                     on_delta=on_delta)

        # 6 · informe
        # 6 · informe. La síntesis es una sola llamada y es la única etapa de
        # razonamiento de la corrida, así que puede permitirse un modelo más
        # capaz aunque sea lento. Los dos no caben en memoria a la vez.
        reporter = model
        if args.reporter != args.vlm:
            state.rows = []
            state.screens_total = 0
            state.reader.reset()
            state.active_model = state.writer_model
            view.stage("redactar", "cambiando al modelo de síntesis",
                       f"{state.reader_model} → {state.writer_model}")
            model.release()
            try:
                reporter = VisionModel(args.reporter)
            except ModelError as exc:
                state.notes.append(f"no se pudo cargar {args.reporter}: {exc}")
                live.stop()
                console.print(error_panel(video.name, str(exc)))
                return False

        state.current_label = ""
        state.screens_total = 0
        state.rows = []
        state.reader.reset()
        state.active_model = state.writer_model
        view.stage("redactar", "redactando el informe",
                   f"sintetizando {len(units)} unidades")

        body, report_usage = write_report(reporter, units, transcript,
                                          on_delta=on_delta)
        total_usage.add(report_usage)

    elapsed = state.elapsed
    md_path = unique_path(DOWNLOADS, video.stem, "md")
    md_path.write_text(
        render_markdown(video, body, units, transcript, duration, elapsed),
        encoding="utf-8")
    json_path = unique_path(DOWNLOADS, video.stem, "json")
    json_path.write_text(
        render_json(video, units, transcript, duration, elapsed,
                    {"reader": args.vlm, "reporter": args.reporter,
                     "whisper": args.whisper},
                    body, total_usage),
        encoding="utf-8")

    console.print(result_panel(video, md_path, json_path, frames_dir,
                               units, duration, elapsed, body, total_usage))
    console.print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analiza una grabación de pantalla cruzando imagen y audio, 100% local.")
    parser.add_argument("--version", action="version",
                        version=f"transcribevideo-mlx {__version__}")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--vlm", default=DEFAULT_MODEL,
                        help="modelo que lee las pantallas (se elige por velocidad: "
                             "es la llamada que se repite)")
    parser.add_argument("--reporter", default=DEFAULT_REPORTER,
                        help="modelo que redacta el informe final (una sola "
                             "llamada; usa el mismo id que --vlm para no cambiar)")
    parser.add_argument("--whisper", choices=list(MODELS), default="large-v3-turbo",
                        help="modelo Whisper para el audio")
    parser.add_argument("--lang", default=None, help="fuerza el idioma (es, en, …)")
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS,
                        help="frecuencia de muestreo para detectar cambios de pantalla")
    parser.add_argument("--threshold", type=int, default=CUT_THRESHOLD,
                        help="distancia dhash (de 1024) para considerar otra pantalla")
    parser.add_argument("--min-screen", type=float, default=MIN_SCREEN_SECONDS,
                        help="segundos que debe durar una pantalla para analizarse; "
                             "los tramos más breves se funden en la anterior")
    parser.add_argument("--max-screens", type=int, default=0,
                        help="tope de pantallas a analizar (0 = sin tope)")
    args = parser.parse_args()

    console.print(banner(args.vlm, args.reporter, args.whisper))

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
