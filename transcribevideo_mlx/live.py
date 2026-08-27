"""Vista viva de la corrida: una aplicación de pantalla completa.

El motor genera JSON token a token, así que se puede mostrar *qué campo está
escribiendo el modelo en este momento* y con qué contenido. Eso convierte la
espera en algo legible: se ve el texto salir de la imagen, línea por línea, en
vez de una barra que avanza sin decir nada.

La distribución es deliberada. A la izquierda el trabajo — hecho, en curso y en
cola — porque responde "¿avanza?". A la derecha lo que el modelo escribe ahora,
porque responde "¿qué está haciendo?". Y abajo el ritmo y el tiempo restante,
que es lo único que importa cuando la respuesta a las dos anteriores es "va bien
pero falta".
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

C1, C2 = "#5AC8FA", "#5E5CE6"
OK = "#30D158"
WARN = "#FF9F0A"
ERR = "#FF453A"
DIM = "grey50"
SPARK = "▁▂▃▄▅▆▇█"
SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
CURSOR = "▋"

#: Etapas de una corrida, en orden. La vista las muestra como recorrido para
#: que se sepa dónde está y cuánto falta del proceso completo, no solo del
#: bucle de pantallas.
STAGES = [
    ("segmentar", "detectando cambios de pantalla"),
    ("oír", "transcribiendo el audio"),
    ("extraer", "extrayendo frames"),
    ("leer", "leyendo las pantallas"),
    ("redactar", "redactando el informe"),
]

#: Nombres legibles de los campos del chunk, en el orden en que el modelo los
#: escribe. Lo literal primero, lo interpretado después.
FIELD_LABELS = {
    "titulo": "título",
    "texto_en_pantalla": "texto leído en pantalla",
    "elementos_ui": "elementos de interfaz",
    "sintesis": "síntesis",
    "se_entiende_sola": "¿se entiende sola?",
    "motivo": "motivo",
}
_FIELD_RE = re.compile(r'"([a-z_]+)"\s*:\s*')


# ── color y formato ─────────────────────────────────────────────────────────
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


# ── lectura incremental del JSON ────────────────────────────────────────────
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


# ── estado ──────────────────────────────────────────────────────────────────
@dataclass
class ScreenRow:
    """Una pantalla en la cola de trabajo, con su estado."""

    index: int
    at: float
    title: str = ""
    status: str = "pending"  # pending | active | done | merged | error

    MARKS = {"done": ("✓", OK), "merged": ("✓", OK), "active": ("▸", C1),
             "error": ("✗", ERR), "pending": ("·", "grey30")}

    def mark(self) -> tuple[str, str]:
        return self.MARKS[self.status]


@dataclass
class RunState:
    """Todo lo que la vista necesita saber de la corrida."""

    name: str = ""
    stage: str = ""
    #: Etapa del recorrido a la que pertenece `stage`. Se lleva aparte porque
    #: hay pasos intermedios (cargar un modelo, intercambiarlo) que no son
    #: etapas propias y no deben mover el indicador hacia adelante.
    stage_key: str = "segmentar"
    detail: str = ""
    started: float = field(default_factory=time.time)

    reader_model: str = ""
    writer_model: str = ""
    active_model: str = ""

    rows: list[ScreenRow] = field(default_factory=list)
    active: int = 0

    screens_done: int = 0
    screens_total: int = 0

    current_label: str = ""
    merging: int = 0

    prompt_tokens: int = 0
    generation_tokens: int = 0
    tps: float = 0.0
    peak_memory: float = 0.0
    tps_history: list[float] = field(default_factory=list)
    #: Segundos que tomó cada unidad ya terminada, para estimar lo que falta.
    unit_times: list[float] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    reader: FieldReader = field(default_factory=FieldReader)
    frame: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def stage_index(self) -> int:
        for i, (short, _full) in enumerate(STAGES):
            if self.stage_key == short:
                return i
        return 0

    def record_tps(self, tps: float) -> None:
        if tps > 0:
            self.tps = tps
            self.tps_history.append(tps)
            del self.tps_history[:-60]

    def record_unit(self, seconds: float) -> None:
        if seconds > 0:
            self.unit_times.append(seconds)

    def eta(self) -> float | None:
        """Segundos restantes del bucle de pantallas.

        Se usa la media de las últimas unidades y no la global: el ritmo cambia
        a lo largo de la corrida (pantallas con más texto tardan más), y una
        estimación que no se corrige es peor que ninguna.
        """
        left = self.screens_total - self.screens_done
        if left <= 0 or not self.unit_times:
            return None
        recent = self.unit_times[-8:]
        return (sum(recent) / len(recent)) * left

    def window(self, size: int) -> list[ScreenRow]:
        """Ventana deslizante alrededor de la pantalla en curso.

        Con videos de doscientas pantallas no cabe la lista entera, y las que
        importan son las recién hechas y las que vienen: eso es lo que dice si
        el trabajo avanza y hacia dónde.
        """
        if not self.rows:
            return []
        ahead = max(2, size // 4)
        start = max(0, min(self.active - (size - ahead), len(self.rows) - size))
        return self.rows[max(0, start):max(0, start) + size]


# ── piezas gráficas ─────────────────────────────────────────────────────────
def spark(values: list[float], width: int) -> Text:
    """Sparkline del ritmo de generación."""
    out = Text()
    if not values or width <= 0:
        return out
    tail = values[-width:]
    top = max(tail) or 1.0
    pad = width - len(tail)
    for _ in range(pad):
        out.append("▁", style="grey27")
    for i, v in enumerate(tail):
        level = int(round((v / top) * (len(SPARK) - 1)))
        pos = (pad + i) / (width - 1) if width > 1 else 0.0
        out.append(SPARK[level], style=interp(C1, C2, pos))
    return out


def bar(done: int, total: int, width: int, frame: int = 0) -> Text:
    """Barra con degradado y un brillo que recorre la parte llena."""
    out = Text()
    if width <= 0:
        return out
    filled = int(round(width * done / total)) if total else 0
    glint = (frame // 2) % max(width, 1)
    for i in range(width):
        pos = i / (width - 1) if width > 1 else 0.0
        if i < filled:
            lit = abs(i - glint) <= 1 and filled > 3
            out.append("█", style=interp(C1, C2, pos) + (" bold" if lit else ""))
        else:
            out.append("─", style="grey23")
    return out


def pulse(frame: int, width: int) -> Text:
    """Onda para las etapas sin progreso medible (carga, transcripción)."""
    out = Text()
    for c in range(max(width, 0)):
        v = (math.sin(c * 0.45 - frame * 0.5) + math.sin(c * 0.27 - frame * 0.31)) / 2
        level = int(round(((v + 1) / 2) * (len(SPARK) - 1)))
        pos = c / (width - 1) if width > 1 else 0.0
        out.append(SPARK[level], style=interp(C1, C2, pos))
    return out


def breadcrumb(state: RunState) -> Text:
    """Las etapas del proceso, con la actual encendida."""
    out = Text(justify="center")
    current = state.stage_index
    for i, (short, _full) in enumerate(STAGES):
        if i < current:
            out.append("● ", style=OK)
            out.append(short, style="grey54")
        elif i == current:
            out.append(SPIN[state.frame % len(SPIN)] + " ", style=C1)
            out.append(short, style=f"bold {C1}")
        else:
            out.append("○ ", style="grey27")
            out.append(short, style="grey30")
        if i < len(STAGES) - 1:
            out.append("   ─   ", style="grey23")
    return out


def _header(state: RunState) -> Panel:
    title = Text()
    title.append("transcribevideo", style=f"bold {C2}")
    title.append("   ", style="grey30")
    title.append(state.name, style="grey74")

    right = Text(justify="right")
    if state.active_model:
        right.append(state.active_model, style="grey42")
        right.append("   ·   ", style="grey27")
    right.append("⏱ ", style=C1)
    right.append(human(state.elapsed), style="grey78")

    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(justify="right")
    top.add_row(title, right)
    return Panel(Group(top, Text(), breadcrumb(state)), box=box.ROUNDED,
                 border_style="grey27", padding=(0, 2))


def _queue(state: RunState, width: int, rows_visible: int) -> Panel:
    """Columna izquierda: qué se hizo, qué se hace y qué falta."""
    rows: list[RenderableType] = []
    for row in state.window(rows_visible):
        glyph, colour = row.mark()
        line = Text(no_wrap=True)
        if row.status == "active":
            # Latido suave para que la fila en curso se distinga sin gritar.
            t = (math.sin(state.frame * 0.25) + 1) / 2
            colour = interp(C1, C2, t)
        line.append(f"{glyph} ", style=colour)
        line.append(f"{clock(row.at)} ", style="grey35")
        room = max(6, width - 12)
        if row.status == "active":
            line.append((row.title or "leyendo…")[:room], style=f"bold {colour}")
        elif row.status == "pending":
            line.append("en cola", style="grey27")
        else:
            style = "grey58" if row.status != "error" else ERR
            line.append((row.title or "—")[:room], style=style)
            if row.status == "merged":
                line.append(" ⇢", style=WARN)
        rows.append(line)

    head = Text()
    head.append(" pantallas ", style="grey42")
    if state.screens_total:
        head.append(f"{state.screens_done}", style="grey85")
        head.append(f"/{state.screens_total} ", style="grey42")
    return Panel(Group(*rows) if rows else Text("—", style="grey27"),
                 box=box.ROUNDED, border_style="grey27", padding=(1, 2),
                 title=head, title_align="left")


def _stream(state: RunState, width: int, rows_visible: int) -> Panel:
    """Columna derecha: lo que el modelo está escribiendo ahora."""
    rows: list[RenderableType] = []
    inner = max(20, width - 6)

    if state.current_label:
        where = Text()
        where.append(state.current_label, style="grey70")
        if state.merging:
            where.append(f"   ⇢ fusionando {state.merging}", style=WARN)
        rows += [where, Text()]

    label = state.reader.label
    if label:
        reading = Text()
        reading.append("leyendo ", style="grey42")
        reading.append(label, style=f"bold {C2}")
        rows += [reading, Text()]
        lines = state.reader.lines(limit=max(3, rows_visible - 5), width=inner - 3)
        for i, line in enumerate(lines):
            body = Text(no_wrap=True)
            body.append("┃ ", style="grey30")
            body.append(line, style="grey82")
            if i == len(lines) - 1 and state.frame // 5 % 2 == 0:
                body.append(CURSOR, style=C1)
            rows.append(body)
    elif not state.rows:
        rows += [Text(), Align.center(pulse(state.frame, min(inner, 46))), Text(),
                 Align.center(Text(state.detail or "trabajando…", style="grey42"))]
    else:
        rows.append(Text("esperando al modelo…", style="grey30"))

    return Panel(Group(*rows), box=box.ROUNDED, border_style="grey27",
                 padding=(1, 2), title=Text(" lectura en vivo ", style="grey42"),
                 title_align="left")


def _footer(state: RunState, width: int) -> Panel:
    inner = max(30, width - 8)

    top = Text()
    if state.screens_total:
        pct = state.screens_done / state.screens_total
        bar_width = max(12, inner - 34)
        top.append_text(bar(state.screens_done, state.screens_total, bar_width,
                            state.frame))
        top.append(f"  {pct * 100:>3.0f}%", style="grey85")
        eta = state.eta()
        if eta:
            top.append("   faltan ", style="grey42")
            top.append(human(eta), style=f"bold {C1}")
    else:
        top.append_text(pulse(state.frame, max(12, inner - 34)))
        if state.detail:
            top.append(f"  {state.detail}", style="grey42")

    stats = Text()
    if state.tps_history:
        stats.append_text(spark(state.tps_history, 22))
        stats.append("  ", style="grey30")
        stats.append(f"{state.tps:.0f}", style="grey85")
        stats.append(" tok/s", style="grey42")
        stats.append("   ", style="grey30")
    stats.append(compact(state.prompt_tokens), style="grey74")
    stats.append(" in", style="grey42")
    stats.append("  ", style="grey30")
    stats.append(compact(state.generation_tokens), style="grey74")
    stats.append(" out", style="grey42")
    if state.peak_memory:
        stats.append("   ", style="grey30")
        stats.append(f"{state.peak_memory:.0f} GB", style="grey74")

    parts: list[RenderableType] = [top, stats]
    if state.notes:
        note = Text(no_wrap=True)
        note.append("! ", style=WARN)
        note.append(state.notes[-1][:inner - 2], style="grey46")
        parts.append(note)
    return Panel(Group(*parts), box=box.ROUNDED, border_style="grey27",
                 padding=(0, 2))


def render(state: RunState, width: int, height: int = 30) -> Layout:
    """Compone la vista completa."""
    footer_height = 5 if state.notes else 4
    body_height = max(6, height - 6 - footer_height)
    rows_visible = max(3, body_height - 4)

    left = max(24, int(width * 0.36))
    right = width - left

    layout = Layout()
    layout.split_column(
        Layout(_header(state), name="head", size=5),
        Layout(name="body", ratio=1),
        Layout(_footer(state, width), name="foot", size=footer_height),
    )
    layout["body"].split_row(
        Layout(_queue(state, left, rows_visible), name="queue", size=left),
        Layout(_stream(state, right, rows_visible), name="stream", ratio=1),
    )
    return layout
