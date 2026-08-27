"""Vista viva de la corrida.

El motor genera JSON token a token, así que se puede mostrar *qué campo está
escribiendo el modelo en este momento* y con qué contenido. Eso convierte la
espera en algo legible: se ve el texto salir de la imagen, campo por campo, en
vez de una barra que avanza sin decir nada.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

C1, C2 = "#5AC8FA", "#5E5CE6"
OK = "#30D158"
WARN = "#FF9F0A"
ERR = "#FF453A"
DIM = "grey50"
SPARK = "▁▂▃▄▅▆▇█"
SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

#: Nombres legibles de los campos del chunk, en el orden en que el modelo los
#: escribe. Lo literal primero, lo interpretado después.
FIELD_LABELS = {
    "titulo": "título",
    "texto_en_pantalla": "texto leído en pantalla",
    "elementos_ui": "elementos de interfaz",
    "narracion": "narración",
    "sintesis": "síntesis",
    "se_entiende_sola": "¿se entiende sola?",
    "motivo": "motivo",
}
_FIELD_RE = re.compile(r'"([a-z_]+)"\s*:\s*')


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
        return f"{hours}h{minutes:02d}m"
    return f"{minutes:02d}:{secs:02d}"


def compact(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n)}"


def clock(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class FieldReader:
    """Sigue qué campo del JSON está escribiendo el modelo, y su contenido.

    No parsea: el JSON está incompleto mientras llega. Solo busca la última
    clave abierta y muestra lo que se acumuló después de ella, que es
    exactamente lo que el modelo está leyendo de la imagen en este instante.
    """

    def __init__(self) -> None:
        self.buffer = ""

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, delta: str) -> None:
        self.buffer += delta

    @property
    def field(self) -> str | None:
        matches = list(_FIELD_RE.finditer(self.buffer))
        return matches[-1].group(1) if matches else None

    @property
    def label(self) -> str | None:
        name = self.field
        if name is None:
            return None
        return FIELD_LABELS.get(name, name)

    def lines(self, limit: int = 5, width: int = 60) -> list[str]:
        """Últimas líneas del valor en curso, ya legibles."""
        matches = list(_FIELD_RE.finditer(self.buffer))
        raw = self.buffer[matches[-1].end():] if matches else self.buffer
        text = (raw.replace('\\n', '\n').replace('\\"', '"')
                   .replace('",', '').replace('"', '').strip())
        out: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            while len(line) > width:
                out.append(line[:width])
                line = line[width:]
            if line:
                out.append(line)
        return out[-limit:]


@dataclass
class RunState:
    """Todo lo que la vista necesita saber de la corrida."""

    name: str = ""
    stage: str = ""
    detail: str = ""
    started: float = field(default_factory=time.time)

    screens_done: int = 0
    screens_total: int = 0

    current_label: str = ""
    merging: int = 0

    prompt_tokens: int = 0
    generation_tokens: int = 0
    tps: float = 0.0
    peak_memory: float = 0.0
    tps_history: list[float] = field(default_factory=list)

    done_titles: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    reader: FieldReader = field(default_factory=FieldReader)
    frame: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def record_tps(self, tps: float) -> None:
        if tps > 0:
            self.tps = tps
            self.tps_history.append(tps)
            del self.tps_history[:-40]


def spark(values: list[float], width: int) -> Text:
    """Sparkline del ritmo de generación."""
    out = Text()
    if not values:
        return out
    tail = values[-width:]
    top = max(tail) or 1.0
    pad = width - len(tail)
    for i in range(pad):
        out.append("▁", style="grey27")
    for i, v in enumerate(tail):
        level = int(round((v / top) * (len(SPARK) - 1)))
        pos = (pad + i) / (width - 1) if width > 1 else 0.0
        out.append(SPARK[level], style=interp(C1, C2, pos))
    return out


def bar(done: int, total: int, width: int) -> Text:
    out = Text()
    filled = int(round(width * done / total)) if total else 0
    for i in range(width):
        pos = i / (width - 1) if width > 1 else 0.0
        out.append("█" if i < filled else "░",
                   style=interp(C1, C2, pos) if i < filled else "grey27")
    return out


def pulse(frame: int, width: int) -> Text:
    """Onda para las etapas sin progreso medible (carga, transcripción)."""
    out = Text()
    for c in range(width):
        v = (math.sin(c * 0.45 - frame * 0.5) + math.sin(c * 0.27 - frame * 0.31)) / 2
        level = int(round(((v + 1) / 2) * (len(SPARK) - 1)))
        pos = c / (width - 1) if width > 1 else 0.0
        out.append(SPARK[level], style=interp(C1, C2, pos))
    return out


def render(state: RunState, width: int) -> Panel:
    inner = max(28, min(width - 14, 62))
    parts: list[RenderableType] = []

    # ── cabecera: etapa y reloj ─────────────────────────────────────────
    head = Text()
    head.append(SPIN[state.frame % len(SPIN)] + "  ", style=C1)
    head.append(state.stage, style="grey78")
    if state.detail:
        head.append(f"  {state.detail}", style=DIM)
    head.append(f"{'':>3}{human(state.elapsed)}", style=DIM)
    parts += [head, Text()]

    # ── progreso ────────────────────────────────────────────────────────
    if state.screens_total:
        counter = Text()
        counter.append(f"{state.screens_done}", style="grey85")
        counter.append(f"/{state.screens_total} pantallas", style=DIM)
        if state.merging:
            counter.append(f"   ⇢ fusionando {state.merging}", style=WARN)
        parts += [bar(state.screens_done, state.screens_total, inner), counter, Text()]
    else:
        parts += [pulse(state.frame, inner), Text()]

    # ── lo que el modelo está leyendo ahora ─────────────────────────────
    if state.current_label:
        where = Text()
        where.append("▸ ", style=C1)
        where.append(state.current_label, style="grey70")
        parts.append(where)

    label = state.reader.label
    if label:
        reading = Text()
        reading.append("  leyendo ", style="grey42")
        reading.append(label, style=f"bold {C2}")
        parts += [Text(), reading]
        for line in state.reader.lines(limit=5, width=inner - 4):
            row = Text()
            row.append("  ┃ ", style="grey35")
            row.append(line, style="grey78")
            parts.append(row)

    # ── tokens ──────────────────────────────────────────────────────────
    if state.tps_history or state.generation_tokens:
        stats = Text()
        stats.append(f"{state.tps:.0f}", style="grey85")
        stats.append(" tok/s", style="grey42")
        stats.append("   ·   ", style="grey30")
        stats.append(compact(state.prompt_tokens), style="grey74")
        stats.append(" in", style="grey42")
        stats.append("  ", style="grey30")
        stats.append(compact(state.generation_tokens), style="grey74")
        stats.append(" out", style="grey42")
        if state.peak_memory:
            stats.append("   ·   ", style="grey30")
            stats.append(f"{state.peak_memory:.1f} GB", style="grey74")
        parts += [Text(), spark(state.tps_history, inner), stats]

    # ── hecho y avisos ──────────────────────────────────────────────────
    if state.done_titles or state.notes:
        parts.append(Rule(style="grey27"))
    for title in state.done_titles[-3:]:
        row = Text()
        row.append("  ✓ ", style=OK)
        row.append(title[:inner], style="grey58")
        parts.append(row)
    for note in state.notes[-2:]:
        row = Text()
        row.append("  ! ", style=WARN)
        row.append(note[:inner], style="grey58")
        parts.append(row)

    return Panel(Group(*parts), box=box.ROUNDED, border_style="grey35",
                 padding=(1, 3), title=Text(f" {state.name} ", style=DIM),
                 title_align="left")
