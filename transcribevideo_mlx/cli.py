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
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    import pyfiglet
except Exception:  # noqa: BLE001 - el wordmark es decorativo
    pyfiglet = None

from . import __version__
from .audio import MODELS, snap_to_speech
from .live import (C1, C2, DIM, ERR, OK, WARN, RunState, ScreenRow, clock,
                   compact, gradient, human, interp, render)
from .report import render_json, render_markdown, unique_path, write_report
from .segment import CUT_THRESHOLD, MIN_SCREEN_SECONDS, SAMPLE_FPS
from .vlm import (DEFAULT_MODEL, DEFAULT_REPORTER, Usage, available_memory_gb,
                  check_headroom, resolve_model_path)

DOWNLOADS = Path.home() / "Downloads"
#: Mínimo entre repintados. El modelo emite decenas de tokens por segundo y
#: redibujar en cada uno gasta más CPU en la UI que en la inferencia.
#:
#: 10 por segundo basta para que el spinner y el cursor se vean fluidos. Cada
#: repintado escribe ~9 KB de pantalla completa y compite por el GIL con el
#: hilo que está generando tokens, así que subirlo se paga en tirones, no en
#: suavidad.
REDRAW_INTERVAL = 0.1

console = Console(file=sys.stdout, highlight=False)


MOTIF = "▚▚▖▘▝▗▚▘▖▝▚▗▘▚▖▝▘▗▚▖▘"
FORMATS = "mp4 · mov · m4v · mkv · webm"


def wordmark() -> str:
    """El logotipo, sin las filas vacías que deja pyfiglet.

    La última fila suele venir llena de espacios, no vacía, así que un
    `rstrip("\\n")` no la quita y el logo queda flotando.
    """
    raw = (pyfiglet.figlet_format("transcribevideo", font="small") if pyfiglet
           else "transcribevideo")
    lines = raw.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def sweep(word: str, t: float) -> Text:
    """El wordmark con un brillo que lo recorre.

    Sirve para la entrada animada: el degradado base se mantiene y solo se
    realza una banda estrecha que viaja, así el arranque tiene vida sin que el
    resultado final cambie de aspecto.
    """
    lines = word.rstrip("\n").split("\n")
    width = max((len(line) for line in lines), default=1) or 1
    head = t * (width + 16) - 8
    out = Text()
    for line in lines:
        for i, char in enumerate(line):
            pos = i / (width - 1) if width > 1 else 0.0
            near = max(0.0, 1.0 - abs(i - head) / 7.0)
            colour = interp(interp(C1, C2, pos), "#FFFFFF", near * 0.75)
            out.append(char, style=colour + (" bold" if near > 0.5 else ""))
        out.append("\n")
    return out


def intro(console_: Console) -> None:
    """Barrido de bienvenida. Solo si hay terminal; dura menos de un segundo."""
    if not console_.is_terminal:
        return
    word = wordmark()
    with Live(console=console_, refresh_per_second=30, transient=True) as live:
        for step in range(22):
            live.update(Group(Text(), Align.center(gradient(MOTIF)),
                              Align.center(sweep(word, step / 21))))
            time.sleep(0.028)


def _hf_cache(model: str) -> Path | None:
    """Carpeta del modelo en la caché de Hugging Face, si está descargado."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = cache / ("models--" + model.replace("/", "--"))
    return folder if folder.is_dir() else None


def _model_status(model: str) -> tuple[bool, float]:
    """¿Está ya en disco, y cuánto pesa?

    Se miran los dos sitios donde puede estar: la carpeta de LM Studio, que es
    de donde lo toma `resolve_model_path`, y la caché de Hugging Face, que es
    donde cae si lo descargó esta herramienta.
    """
    path = Path(resolve_model_path(model))
    if not path.is_dir():
        cached = _hf_cache(model)
        if cached is None:
            return False, 0.0
        weights = list(cached.rglob("*.safetensors")) + list(cached.rglob("*.gguf"))
        return True, sum(f.stat().st_size for f in weights if f.is_file()) / 1e9
    weights = list(path.glob("*.safetensors")) or list(path.glob("*.gguf"))
    return True, sum(f.stat().st_size for f in weights) / 1e9


def ensure_models(models: list[str], console_: Console) -> bool:
    """Descarga lo que falte antes de entrar a la vista de pantalla completa.

    `mlx_vlm.load` descargaría igual, pero lo haría ya dentro del TUI, donde las
    barras de progreso de Hugging Face quedan ocultas o rompen el panel. Bajar
    dieciséis gigas sin ninguna señal es indistinguible de un cuelgue, así que
    esto se hace afuera y con su progreso a la vista.
    """
    missing = [m for m in dict.fromkeys(models) if not _model_status(m)[0]]
    if not missing:
        return True

    console_.print()
    console_.print(Align.center(Panel(
        Text.assemble(
            ("Primera corrida: falta descargar ", "grey74"),
            (f"{len(missing)} modelo{'s' if len(missing) > 1 else ''}", f"bold {WARN}"),
            (".\nSon varios gigas y queda en caché para siempre.", "grey74")),
        box=box.ROUNDED, border_style=WARN, padding=(1, 3),
        width=card_width(console_.width))))
    console_.print()

    for model in missing:
        console_.print(Text.assemble(("  ↓  ", WARN), (model, "grey74")))
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(model)
        except Exception as exc:  # noqa: BLE001 - se le muestra al usuario
            console_.print(error_panel(model, f"No se pudo descargar:\n{exc}"))
            return False
    console_.print()
    return True


def _whisper_ready(model: str) -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return any(cache.glob(f"models--mlx-community--whisper-{model}*"))


def card_width(width: int) -> int:
    """Ancho común de las tarjetas de la portada.

    Compartirlo es lo que hace que la pantalla se lea como una composición y no
    como dos cajas sueltas de tamaños distintos.
    """
    return max(56, min(width - 8, 74))


def pipeline_card(vlm: str, reporter: str, whisper: str, width: int) -> Panel:
    """Las cuatro etapas y con qué se hace cada una.

    Muestra si cada pieza ya está en disco: enterarse de que faltan 32 GB por
    descargar vale mucho más antes de arrancar que a los cinco minutos.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="grey30", width=2)
    table.add_column(width=22)
    table.add_column(ratio=1)
    table.add_column(justify="right", width=10)

    def row(num: str, what: str, who: str, state: Text) -> None:
        table.add_row(Text(num, style=C2), Text(what, style="grey74"),
                      Text(who, style="grey50"), state)

    def ready(ok: bool, size: float, missing: str) -> Text:
        state = Text()
        if ok:
            state.append("✓ listo", style=OK)
        else:
            state.append(missing if not size else f"↓ {size:.0f} GB", style=WARN)
        return state

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    reader_ok, reader_size = _model_status(vlm)
    writer_ok, writer_size = _model_status(reporter)

    row("①", "detecta pantallas", "dhash 1024 bits · ffmpeg",
        ready(ffmpeg_ok, 0, "✗ instala ffmpeg"))
    row("②", "transcribe el audio", f"whisper {whisper}",
        ready(_whisper_ready(whisper), 1.5, "↓ descarga"))
    row("③", "lee cada pantalla", _short_model(vlm),
        ready(reader_ok, reader_size or 16, "↓ descarga"))
    row("④", "redacta el informe", _short_model(reporter),
        ready(writer_ok, writer_size or 16, "↓ descarga"))

    # La memoria es la única de estas condiciones que cambia entre una corrida y
    # la siguiente. Sin ella el sistema no da un error: se congela.
    libre = available_memory_gb()
    necesita = max(reader_size, writer_size) * 1.35 or 21.0
    holgura = Text()
    if libre >= necesita:
        holgura.append(f"✓ {libre:.0f} GB libres", style=OK)
    else:
        holgura.append(f"⚠ solo {libre:.0f} GB", style=WARN)
    table.add_row(Text("⑤", style=C2), Text("memoria", style="grey74"),
                  Text(f"~{necesita:.0f} GB en uso", style="grey50"),
                  holgura)

    panel = Panel(table, box=box.ROUNDED, border_style="grey27", padding=(1, 2),
                  width=width, title=Text(" el motor ", style="grey42"),
                  title_align="left")
    if not ffmpeg_ok:
        # Sin ffmpeg no hay nada que hacer, y el error saldría recién al abrir
        # el video: vale más decirlo en la portada.
        panel.border_style = ERR
    return panel


def welcome(vlm: str, reporter: str, whisper: str, width: int) -> Group:
    tagline = Text(justify="center")
    tagline.append("convierte una grabación de pantalla en un informe", style="grey62")
    tagline.append("   ·   ", style="grey27")
    tagline.append("100% local", style=OK)

    return Group(
        Text(), Align.center(gradient(MOTIF)),
        Align.center(gradient(wordmark().rstrip("\n"))),
        tagline, Text(),
        Align.center(pipeline_card(vlm, reporter, whisper, card_width(width))),
        Text(),
    )


def dropzone(width: int) -> Group:
    """La zona donde se suelta el archivo."""
    invite = Text(justify="center")
    invite.append("▶  ", style=C1)
    invite.append("arrastra el video aquí y presiona Enter", style=f"bold {C1}")

    hint = Text(justify="center")
    hint.append(FORMATS, style="grey35")
    hint.append("        ", style="grey27")
    hint.append("salida en ", style="grey35")
    hint.append("~/Downloads", style="grey46")

    return Group(
        Align.center(Panel(invite, box=box.ROUNDED, border_style=C2,
                           padding=(1, 2), width=card_width(width))),
        Text(), hint, Text())


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
        self.state.stage_started = time.time()
        self.draw(force=True)

    def work(self, fn):
        """Ejecuta algo bloqueante sin que la vista se congele.

        Cargar un modelo o transcribir el audio no devuelve el control hasta
        terminar. Rich sigue repintando, pero siempre el mismo cuadro: la
        animación existe y está quieta, que es peor que no tenerla, porque
        justo en las esperas largas parece que el programa colgó. Se hace el
        trabajo en un hilo y se anima desde aquí.
        """
        result: dict = {}

        def run() -> None:
            try:
                result["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - se relanza abajo
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            self.draw(force=True)
            time.sleep(REDRAW_INTERVAL)
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def draw(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < REDRAW_INTERVAL:
            return
        self._last = now
        self.state.frame += 1
        # refresh explícito: con auto_refresh apagado, update() por sí solo no
        # pinta nada.
        self.live.update(render(self.state, self.width, self.height),
                         refresh=True)


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


def context_tokens(units, transcript) -> int:
    """Tamaño aproximado del prompt del informe, en tokens.

    Es una estimación por caracteres, no un conteo real: sirve para decirle al
    usuario por qué el primer token tarda, no para presupuestar.
    """
    chars = sum(len(str(u.chunk)) for u in units) + len(transcript.full_text)
    return chars // 4


def run_summary(state: RunState, whisper: str, duration: float, raw_cuts: int,
                cuts: int, transcript, units, usage: Usage) -> list:
    """Las cifras de la corrida, para la columna izquierda durante el informe."""
    merges = sum(1 for u in units if u.merged)
    errors = sum(1 for u in units if u.error)
    per_screen = (sum(state.unit_times) / len(state.unit_times)
                  if state.unit_times else 0.0)
    words = len(transcript.full_text.split())
    return [
        ("video", [("duración", clock(duration)),
                   ("pantallas", f"{raw_cuts} → {cuts}"),
                   ("fundidas", raw_cuts - cuts)]),
        ("audio", [("idioma", transcript.language),
                   ("segmentos", len(transcript.utterances)),
                   ("palabras", f"{words:,}".replace(",", "."))]),
        ("lectura", [("unidades", len(units)),
                     ("fusiones", merges),
                     ("errores", errors),
                     ("por pantalla", f"{per_screen:.1f} s"),
                     ("tokens", f"{compact(usage.prompt_tokens)} / "
                                f"{compact(usage.generation_tokens)}")]),
        ("modelos", [("lee", state.reader_model),
                     ("redacta", state.writer_model),
                     ("oye", whisper)]),
    ]


def process(video: Path, args) -> Path | None:
    """Procesa un video. Devuelve la ruta del informe, o None si falló."""
    from . import audio as audio_mod
    from . import pipeline as pipeline_mod
    from . import segment as segment_mod
    from .vlm import ModelError, VisionModel

    if not video.exists():
        console.print(error_panel(video.name, f"No existe el archivo:\n{video}"))
        return None

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

    # `auto_refresh` apagado a propósito: con él, Rich levanta un hilo que
    # repinta por su cuenta el MISMO contenido, y sumado a los repintados de
    # `View.draw` daban ~32 pantallas completas por segundo —unos 290 KB/s al
    # terminal— compitiendo por el GIL con el hilo que genera tokens. Ese exceso
    # se veía como tirones, no como fluidez. Aquí pinta solo quien tiene algo
    # nuevo que mostrar.
    with Live(console=console, screen=full_screen, auto_refresh=False,
              transient=not full_screen) as live:
        view = View(live, state, console.width, height)

        # 1 · segmentar
        view.stage("segmentar", "detectando cambios de pantalla")
        duration = segment_mod.probe_duration(video)
        state.detail = f"{clock(duration)} de video"
        view.draw(force=True)

        hashes = view.work(lambda: segment_mod.sample_hashes(video, args.fps))
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
        # Antes de transcribir hay que cargar el modelo, y eso tarda. Decir
        # "transcribiendo" mientras todavía carga es mentir sobre en qué se está
        # yendo el tiempo; la etiqueta cambia sola con el primer segmento.
        view.stage("oír", "cargando whisper", args.whisper)
        state.free_stream = True

        def on_segment(line: str) -> None:
            if state.stage != "transcribiendo el audio":
                state.stage = "transcribiendo el audio"
                state.detail = args.whisper
            state.reader.feed(line + "\n")

        transcript = view.work(lambda: audio_mod.transcribe(
            video, args.whisper, args.lang, on_segment=on_segment))
        state.free_stream = False
        state.reader.reset()
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
        screens = view.work(lambda: segment_mod.extract_screens(
            video, cuts, duration, frames_dir,
            on_skip=lambda at, why: state.notes.append(
                f"pantalla en {clock(at)} descartada, no se pudo extraer")))
        if not screens:
            live.stop()
            console.print(error_panel(video.name, "No se pudo extraer ninguna pantalla."))
            return None

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
        if aviso := check_headroom(args.vlm, size):
            live.stop()
            console.print(error_panel("memoria insuficiente", aviso))
            return None

        try:
            model = view.work(lambda: VisionModel(args.vlm))
        except ModelError as exc:
            live.stop()
            console.print(error_panel(video.name, str(exc)))
            return None

        # 5 · analizar pantalla por pantalla
        state.screens_total = len(screens)
        state.rows = [ScreenRow(index=i, at=s.start) for i, s in enumerate(screens)]
        view.stage("leer", "leyendo las pantallas", f"{len(screens)} pantallas")

        # Estos callbacks corren en el hilo trabajador: solo mutan estado, nunca
        # dibujan. Quien dibuja es el bucle de `view.work`, en el hilo principal.
        # Si dibujaran aquí, la vista se congelaría durante los prefill —cuando
        # el modelo lee la imagen o las 43 unidades del informe y todavía no
        # emite ningún token—, que es justo la espera más larga de cada llamada.
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

        def on_delta(delta: str) -> None:
            state.reader.feed(delta)
            # El título aparece primero en el JSON: en cuanto se lee, la fila
            # activa deja de decir "leyendo…" y muestra de qué pantalla se trata.
            if state.reader.field == "titulo" and state.rows:
                lines = state.reader.lines(limit=1, width=60)
                if lines:
                    state.rows[state.active].title = lines[0]

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

        units = view.work(lambda: pipeline_mod.analyze(
            model, screens, transcript,
            on_call=on_call, on_unit=on_unit, on_delta=on_delta))

        # 6 · informe. La síntesis es una sola llamada y es la única etapa de
        # razonamiento de la corrida, así que puede permitirse un modelo más
        # capaz aunque sea lento. Los dos no caben en memoria a la vez.
        state.rows = []
        state.screens_total = 0
        state.reader.reset()
        state.summary = run_summary(state, args.whisper, duration, len(raw_cuts),
                                    len(cuts), transcript, units, total_usage)

        reporter = model
        if args.reporter != args.vlm:
            # Dos pasos con nombre propio en vez de un "cambiando de modelo"
            # genérico: liberar 15 GB y cargar otros 16 son esperas distintas y
            # conviene ver en cuál se está.
            state.active_model = state.writer_model
            view.stage("redactar", f"liberando {state.reader_model}",
                       "los dos modelos no caben en memoria")
            view.work(model.release)

            size = model_size_gb(args.reporter)
            view.stage("redactar", f"cargando {state.writer_model}",
                       f"{size:.1f} GB desde disco" if size else "desde disco")
            if aviso := check_headroom(args.reporter, size):
                live.stop()
                console.print(error_panel("memoria insuficiente", aviso))
                return None

            try:
                reporter = view.work(lambda: VisionModel(args.reporter))
            except ModelError as exc:
                state.notes.append(f"no se pudo cargar {args.reporter}: {exc}")
                live.stop()
                console.print(error_panel(video.name, str(exc)))
                return None

        state.current_label = ""
        state.screens_total = 0
        state.rows = []
        state.reader.reset()
        state.free_stream = True     # el informe es Markdown, no JSON
        state.active_model = state.writer_model
        state.summary = run_summary(state, args.whisper, duration, len(raw_cuts),
                                    len(cuts), transcript, units, total_usage)

        # El prefill del informe son decenas de miles de tokens y durante ese
        # rato no llega ninguno: decir cuántos son evita que parezca detenido.
        contexto = context_tokens(units, transcript)
        view.stage("redactar", "leyendo las unidades",
                   f"contexto de ~{compact(contexto)} tokens")

        def on_report_delta(delta: str) -> None:
            if state.stage != "redactando el informe":
                state.stage = "redactando el informe"
                state.detail = f"{len(units)} unidades"
            state.reader.feed(delta)

        body, report_usage = view.work(lambda: write_report(
            reporter, units, transcript, on_delta=on_report_delta))
        state.free_stream = False
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
    return md_path


def actions_bar(report: Path | None) -> Group:
    """Qué se puede hacer al terminar, sin volver a arrancar el programa."""
    bar = Text(justify="center")
    if report:
        bar.append("  Enter ", style=f"bold {OK}")
        bar.append("abrir el informe", style="grey62")
        bar.append("      ", style="grey27")
    bar.append("  o ", style=f"bold {C1}")
    bar.append("procesar otro video", style="grey62")
    bar.append("      ", style="grey27")
    bar.append("  q ", style="grey54")
    bar.append("salir", style="grey62")
    return Group(Align.center(Panel(bar, box=box.ROUNDED, border_style="grey30",
                                    padding=(0, 2), width=card_width(console.width))),
                 Text())


def after_run(report: Path | None) -> list[str] | None:
    """Menú posterior a una corrida.

    Devuelve rutas nuevas si se pide procesar otro video, o None para salir.
    Terminar y cerrarse deja al usuario con la terminal limpia y el informe
    perdido en el scrollback; es más útil quedarse y ofrecer qué hacer.
    """
    while True:
        console.print(actions_bar(report))
        try:
            choice = console.input("   ➜  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None

        if choice in ("q", "salir", "exit"):
            return None
        if choice in ("", "a", "abrir") and report:
            subprocess.run(["open", str(report)], check=False)
            continue
        if choice in ("o", "otro"):
            console.print(dropzone(console.width))
            try:
                files = shlex.split(console.input("   ➜  "))
            except (EOFError, KeyboardInterrupt):
                console.print()
                return None
            if files:
                return files
            continue
        # Cualquier otra cosa que parezca una ruta se toma como el video.
        if choice:
            return shlex.split(choice)


def main() -> int:
    # Las librerías de abajo emiten avisos por stderr en mitad de la corrida
    # (mel filters de transformers, deprecaciones de torch). Bajo la vista de
    # pantalla completa se escriben encima del panel y lo dejan ilegible, y no
    # son accionables para quien usa la herramienta.
    warnings.filterwarnings("ignore")

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

    files = args.files
    if not files:
        intro(console)
    console.print(welcome(args.vlm, args.reporter, args.whisper, console.width))

    if not files:
        console.print(dropzone(console.width))
        try:
            files = shlex.split(console.input("   ➜  "))
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 1
        console.print()

    paths = [Path(os.path.expanduser(f)).resolve() for f in files if f.strip()]
    if not paths:
        console.print(error_panel("entrada", "No se indicó ningún video."))
        return 1

    if shutil.which("ffmpeg") is None:
        console.print(error_panel(
            "ffmpeg", "Falta ffmpeg, que es lo que decodifica el video.\n"
                      "Instálalo con:  brew install ffmpeg"))
        return 1

    if not ensure_models([args.vlm, args.reporter], console):
        return 1

    interactive = console.is_terminal and not args.files
    done = failed = 0
    last_report: Path | None = None

    while paths:
        for path in paths:
            try:
                report = process(path, args)
            except KeyboardInterrupt:
                console.print()
                console.print(error_panel(path.name, "Interrumpido."))
                return 1
            except Exception as exc:  # noqa: BLE001 - un video roto no tumba el lote
                console.print(error_panel(path.name, f"{type(exc).__name__}: {exc}"))
                report = None
            if report:
                done += 1
                last_report = report
            else:
                failed += 1

        if len(paths) > 1:
            console.print(Align.center(
                Text(f"listo · {done} procesado(s), {failed} con problema",
                     style=DIM)))

        # Al terminar no se cierra: queda a la vista qué salió y qué se puede
        # hacer con ello. Sin terminal (salida redirigida) no hay a quién
        # preguntarle, así que se sale como cualquier comando.
        if not interactive:
            break
        siguiente = after_run(last_report)
        if not siguiente:
            break
        paths = [Path(os.path.expanduser(f)).resolve() for f in siguiente if f.strip()]

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
