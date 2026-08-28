"""Salida NDJSON: el contrato que consume cualquier front que no sea el TUI."""
import io
import json

from transcribevideo_mlx.events import EventStream, EventView
from transcribevideo_mlx.live import RunState


def stream():
    buf = io.StringIO()
    return buf, EventStream(buf)


def lineas(buf):
    return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]


def test_one_json_object_per_line():
    buf, s = stream()
    s.emit("stage", key="oír", label="cargando whisper")
    s.emit("screen_done", index=0, title="Ajustes")
    assert len(lineas(buf)) == 2


def test_every_event_carries_its_type_and_time():
    buf, s = stream()
    s.emit("finished", elapsed=12.3)
    evento = lineas(buf)[0]
    assert evento["t"] == "finished"
    assert isinstance(evento["ts"], float)
    assert evento["elapsed"] == 12.3


def test_accents_survive_the_wire():
    """El contenido es español: escaparlo lo volvería ilegible al depurar."""
    buf, s = stream()
    s.emit("screen_done", title="Configuración de sesión")
    assert "Configuración de sesión" in buf.getvalue()
    assert lineas(buf)[0]["title"] == "Configuración de sesión"


def test_a_newline_inside_a_field_does_not_break_the_protocol():
    """El texto de pantalla trae saltos de línea; el protocolo es por línea."""
    buf, s = stream()
    s.emit("screen_done", text="linea uno\nlinea dos\nlinea tres")
    assert len(buf.getvalue().splitlines()) == 1
    assert lineas(buf)[0]["text"].count("\n") == 2


def test_the_view_reports_stages_and_keeps_state():
    buf, s = stream()
    state = RunState()
    view = EventView(state, s)
    view.stage("leer", "leyendo las pantallas", "41 pantallas")
    assert state.stage_key == "leer"
    assert lineas(buf)[0] == {**lineas(buf)[0], "t": "stage", "key": "leer",
                              "label": "leyendo las pantallas",
                              "detail": "41 pantallas"}


def test_the_view_runs_work_without_animating():
    buf, s = stream()
    view = EventView(RunState(), s)
    assert view.work(lambda: 7) == 7
    view.draw(force=True)          # no debe hacer nada ni fallar
    assert lineas(buf) == []
