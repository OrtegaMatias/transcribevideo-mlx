"""Informe de respaldo: qué se entrega cuando la síntesis no se puede hacer."""
from pathlib import Path

from transcribevideo_mlx.pipeline import Unit
from transcribevideo_mlx.report import fallback_body
from transcribevideo_mlx.segment import Screen


def unidades(n=3):
    out = []
    for i in range(n):
        s = [Screen(index=i, start=i * 30.0, end=(i + 1) * 30.0,
                    frame=Path(f"s{i}.png"))]
        out.append(Unit(index=i, screens=s, start=i * 30.0, end=(i + 1) * 30.0,
                        chunk={"titulo": f"Pantalla {i}",
                               "texto_en_pantalla": f"contenido {i}"}))
    return out


def test_the_expensive_work_is_never_thrown_away():
    """Leer las pantallas es la parte cara; si falla la síntesis, se entrega igual."""
    cuerpo = fallback_body(unidades(3), "Sin memoria para el redactor.")
    for i in range(3):
        assert f"Pantalla {i}" in cuerpo


def test_it_says_plainly_what_is_missing():
    cuerpo = fallback_body(unidades(2), "Sin memoria para el redactor.")
    assert "sin síntesis" in cuerpo.lower()
    assert "Sin memoria para el redactor." in cuerpo


def test_it_points_at_the_json_for_regenerating():
    """El JSON permite rehacer el informe sin reprocesar el video."""
    assert ".json" in fallback_body(unidades(1), "motivo")


def test_units_that_failed_are_not_listed_as_content():
    us = unidades(2)
    us[0].error = "boom"
    cuerpo = fallback_body(us, "motivo")
    assert "Pantalla 1" in cuerpo
    assert "Pantalla 0" not in cuerpo


def test_it_works_with_no_units_at_all():
    assert fallback_body([], "motivo")
