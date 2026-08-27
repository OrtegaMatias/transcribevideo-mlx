"""Lectura incremental del JSON que el modelo va escribiendo."""
from transcribevideo_mlx.live import (FieldReader, RunState, ScreenRow, compact,
                                      render, spark)


def feed(*deltas: str) -> FieldReader:
    reader = FieldReader()
    for delta in deltas:
        reader.feed(delta)
    return reader


def test_no_field_before_the_first_key():
    assert feed("{", "\n  ").field is None


def test_current_field_is_the_last_key_opened():
    reader = feed('{"titulo": "Menú', ' Principal", "texto_en_pantalla": "RUT')
    assert reader.field == "texto_en_pantalla"
    assert reader.label == "texto leído en pantalla"


def test_value_accumulates_as_it_streams():
    reader = feed('{"titulo": "Gestión')
    assert reader.lines() == ["Gestión"]
    reader.feed(' de Usuarios"')
    assert reader.lines() == ["Gestión de Usuarios"]


def test_escaped_newlines_become_real_lines():
    reader = feed('{"texto_en_pantalla": "RUT: 1-9\\nNombre: Ana')
    assert reader.lines() == ["RUT: 1-9", "Nombre: Ana"]


def test_only_the_last_lines_are_shown():
    body = "\\n".join(f"linea {i}" for i in range(20))
    reader = feed(f'{{"texto_en_pantalla": "{body}')
    lines = reader.lines(limit=3)
    assert lines == ["linea 17", "linea 18", "linea 19"]


def test_long_lines_are_wrapped_not_truncated():
    reader = feed('{"texto_en_pantalla": "' + "x" * 50)
    assert reader.lines(limit=5, width=20) == ["x" * 20, "x" * 20, "x" * 10]


def test_unknown_key_falls_back_to_its_own_name():
    assert feed('{"campo_raro": "v').label == "campo_raro"


def test_reset_clears_the_buffer():
    reader = feed('{"titulo": "algo')
    reader.reset()
    assert reader.field is None
    assert reader.lines() == []


def test_compact_numbers():
    assert compact(950) == "950"
    assert compact(1500) == "1.5k"
    assert compact(2_400_000) == "2.4M"


def test_spark_is_empty_without_data():
    assert spark([], 10).plain == ""


def test_spark_fits_the_requested_width():
    assert len(spark([1.0, 5.0, 3.0], 10).plain) == 10


# ── ventana deslizante de pantallas ─────────────────────────────────────

def state_with(count: int, active: int) -> RunState:
    st = RunState(rows=[ScreenRow(index=i, at=i * 10.0) for i in range(count)],
                  active=active, screens_total=count, screens_done=active)
    for row in st.rows[:active]:
        row.status = "done"
    st.rows[active].status = "active"
    return st


def test_window_never_exceeds_its_size():
    assert len(state_with(200, 90).window(9)) == 9


def test_short_runs_show_every_screen():
    assert len(state_with(4, 1).window(9)) == 4


def test_window_keeps_the_active_screen_visible():
    for active in (0, 1, 57, 198, 199):
        st = state_with(200, active)
        assert st.rows[active] in st.window(9)


def test_window_shows_what_is_still_queued():
    """La cola es la mitad del valor: dice hacia dónde va el trabajo."""
    st = state_with(200, 90)
    assert any(r.status == "pending" for r in st.window(9))


def test_window_at_the_end_does_not_run_off():
    st = state_with(20, 19)
    win = st.window(9)
    assert len(win) == 9
    assert win[-1].index == 19


def test_window_is_empty_before_screens_exist():
    assert RunState().window(9) == []


def test_render_survives_every_stage():
    """El panel se dibuja igual sin pantallas, con ellas y al redactar."""
    for st in (RunState(name="x", stage="cargando"),
               state_with(50, 10),
               RunState(name="x", stage="redactando", generation_tokens=900)):
        st.notes.append("un aviso")
        assert render(st, 100) is not None
