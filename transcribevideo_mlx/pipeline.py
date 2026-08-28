"""Orquestación: de pantallas y transcripción a unidades de información.

Aquí vive el mecanismo de continuidad. Una pantalla puede cambiar a mitad de
una explicación, y analizar ese tramo por separado produce dos mitades que no
se entienden solas. El modelo juzga si su tramo está completo; si declara que
no, el harness descarta ese análisis y vuelve a llamar fusionando la pantalla
siguiente, hasta que la unidad cierre o hasta tocar el tope.

Tres salvaguardas evitan que eso degenere:

- **Una sola dirección.** Solo se pregunta si la idea sigue hacia adelante.
  "Me falta lo anterior" y "mi idea continúa" son el mismo evento visto desde
  los dos lados, así que mirar solo hacia adelante cubre ambos casos sin tener
  que rehacer una unidad ya emitida.
- **Tope duro.** Una unidad no crece más allá de MAX_SCREENS_PER_UNIT. Sin él,
  una cadena A→B→C fusionaría el video entero en una sola llamada.
- **Degradación, no fallo.** Al llegar al tope se marca `enlace_pendiente` y el
  informe final cose la idea, que tiene todos los tramos a la vista.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .audio import Transcript
from .prompts import (CHUNK_SYSTEM, CHUNK_SYSTEM_OCR, chunk_user_prompt,
                      chunk_user_prompt_ocr)
from .segment import Screen
from .vlm import ModelError, Usage, VisionModel

#: Máximo de pantallas que se pueden fusionar en una unidad.
MAX_SCREENS_PER_UNIT = 3
#: Segundos de audio vecino que se muestran como contexto no analizable, para
#: que el modelo pueda juzgar si su tramo quedó truncado.
PREVIEW_SECONDS = 15.0


@dataclass
class Unit:
    """Un tramo analizado: una o más pantallas y su información extraída."""

    index: int
    screens: list[Screen]
    start: float
    end: float
    chunk: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    #: Segundos que costó el OCR nativo, cuando se usa. Se lleva aparte
    #: porque no es tiempo de modelo y mezclarlo escondería la comparación.
    ocr_seconds: float = 0.0
    link_pending: bool = False
    error: str | None = None

    @property
    def merged(self) -> bool:
        return len(self.screens) > 1

    @property
    def title(self) -> str:
        """Nombre de la unidad, con respaldo si el modelo no puso ninguno.

        El prompt exige un título siempre, pero una pantalla sin texto visible
        sigue tentando al modelo a dejarlo vacío. Antes de rendirse a un
        "(sin título)" que no dice nada, se usa la primera línea de lo que se
        leyó, que casi siempre identifica la pantalla igual de bien.
        """
        if titulo := (self.chunk.get("titulo") or "").strip():
            return titulo
        screen_text = (self.chunk.get("texto_en_pantalla") or "").strip()
        if first_line := next((l.strip() for l in screen_text.splitlines()
                               if l.strip()), ""):
            return first_line[:60]
        return "pantalla sin texto"


def analyze(model: VisionModel, screens: list[Screen], transcript: Transcript,
            on_call: Callable[[int, int], None] | None = None,
            on_unit: Callable[[Unit], None] | None = None,
            on_delta: Callable[[str], None] | None = None,
            use_ocr: bool = False) -> list[Unit]:
    """Convierte las pantallas en unidades de información.

    `on_call` se invoca antes de cada llamada al modelo con (pantalla_inicial,
    pantallas_en_la_unidad) para que la UI muestre avance real, incluidas las
    llamadas extra que consume una fusión. `on_delta` recibe cada fragmento de
    texto que el modelo va generando.
    """
    units: list[Unit] = []
    start_index = 0

    while start_index < len(screens):
        span, unit = 1, None
        # Una fusión tira a la basura el análisis previo, pero esos tokens se
        # gastaron igual. Se acumulan para no subreportar lo que costó la
        # corrida.
        spent = Usage()

        while True:
            group = screens[start_index:start_index + span]
            if on_call:
                on_call(start_index, span)

            unit = _analyze_group(model, len(units), group, transcript,
                                  on_delta, use_ocr)
            spent.add(unit.usage)

            complete = bool(unit.chunk.get("se_entiende_sola", True))
            at_cap = span >= MAX_SCREENS_PER_UNIT
            no_more = start_index + span >= len(screens)

            if unit.error or complete:
                break
            if at_cap or no_more:
                unit.link_pending = True
                break
            span += 1  # reintenta la unidad fusionando la pantalla siguiente

        unit.usage = spent
        units.append(unit)
        if on_unit:
            on_unit(unit)
        start_index += span

    return units


def _analyze_group(model: VisionModel, index: int, group: list[Screen],
                   transcript: Transcript,
                   on_delta: Callable[[str], None] | None = None,
                   use_ocr: bool = False) -> Unit:
    start, end = group[0].start, group[-1].end
    unit = Unit(index=index, screens=list(group), start=start, end=end)

    ventana = dict(
        window_label=f"{_ts(start)} → {_ts(end)}",
        window_audio=transcript.stamped_between(start, end),
        previous_tail=transcript.stamped_between(
            max(0.0, start - PREVIEW_SECONDS), start),
        next_head=transcript.stamped_between(end, end + PREVIEW_SECONDS),
        n_screens=len(group),
    )

    screen_text = ""
    if use_ocr:
        import time as _time

        from . import ocr
        t0 = _time.time()
        screen_text = ocr.read_screens([s.frame for s in group])
        unit.ocr_seconds = _time.time() - t0
        system = CHUNK_SYSTEM_OCR
        prompt = chunk_user_prompt_ocr(screen_text=screen_text, **ventana)
    else:
        system = CHUNK_SYSTEM
        prompt = chunk_user_prompt(**ventana)

    try:
        unit.chunk, unit.usage = model.analyze_screen(
            system, prompt, [s.frame for s in group], on_delta)
        if use_ocr:
            # El texto literal lo pone el OCR, no el modelo: así el campo lo
            # genera algo que nunca oyó el audio y la contaminación deja de ser
            # posible por construcción, no solo improbable por prompt.
            unit.chunk["texto_en_pantalla"] = screen_text
    except ModelError as exc:
        # Un tramo ilegible no debe costar la corrida entera: se marca y el
        # informe final lo verá como un hueco declarado.
        unit.error = str(exc)
        unit.chunk = {}
    return unit


def _ts(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
