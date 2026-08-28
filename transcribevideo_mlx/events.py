"""Salida legible por máquina: una línea JSON por evento (NDJSON).

Existe para que otro front —la app de escritorio, un servicio, un script— pueda
seguir una corrida sin parsear el TUI. Esa salida está hecha para humanos: tiene
colores, se redibuja sobre sí misma y cambia con cada ajuste de diseño. Cualquier
cosa que dependiera de ella se rompería en el siguiente commit.

El contrato es deliberadamente plano: un objeto por línea, con `t` como tipo de
evento. Un consumidor puede ignorar los tipos que no conoce, y agregar tipos
nuevos no rompe a nadie.
"""
from __future__ import annotations

import json
import sys
import time


class EventStream:
    """Escribe eventos NDJSON. Cada línea se vacía al momento.

    El vaciado inmediato no es un detalle: quien lee del otro lado lo hace línea
    a línea, y con búfer los eventos llegarían en ráfagas al final, que es
    exactamente lo contrario de lo que sirve para mostrar progreso.
    """

    def __init__(self, out=None):
        self._out = out or sys.stdout

    def emit(self, kind: str, **data) -> None:
        payload = {"t": kind, "ts": round(time.time(), 3), **data}
        self._out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._out.flush()


class EventView:
    """Reemplaza a la vista viva cuando la corrida la consume una máquina.

    Expone la misma interfaz mínima que usa `process` —`stage`, `work`, `draw`,
    `event`— para que el motor no tenga que saber quién lo está mirando.
    """

    def __init__(self, state, stream: EventStream):
        self.state = state
        self.stream = stream

    def stage(self, key: str, label: str, detail: str = "") -> None:
        self.state.stage_key = key
        self.state.stage = label
        self.state.detail = detail
        self.state.stage_started = time.time()
        self.stream.emit("stage", key=key, label=label, detail=detail)

    def event(self, kind: str, **data) -> None:
        self.stream.emit(kind, **data)

    def work(self, fn):
        """Sin animación que mantener, el trabajo corre directo."""
        return fn()

    def draw(self, force: bool = False) -> None:
        """No hay nada que dibujar."""
