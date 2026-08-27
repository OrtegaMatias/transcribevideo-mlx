"""El mecanismo de continuidad, con un modelo falso.

Es la lógica más delicada del motor y la más cara de probar en vivo, así que se
ejercita contra un doble que devuelve veredictos guionados.
"""
from pathlib import Path

from transcribevideo_mlx.audio import Transcript, Utterance
from transcribevideo_mlx.pipeline import MAX_SCREENS_PER_UNIT, analyze
from transcribevideo_mlx.segment import Screen
from transcribevideo_mlx.vlm import ModelError


class FakeModel:
    """Responde `se_entiende_sola` según cuántas pantallas recibe.

    `complete_at[n]` dice si una unidad de n pantallas se considera completa.
    """

    def __init__(self, complete_at: dict[int, bool], fail_on: set[int] | None = None):
        self.complete_at = complete_at
        self.fail_on = fail_on or set()
        self.calls: list[int] = []

    def analyze_screen(self, system, user, images):
        self.calls.append(len(images))
        if len(images) in self.fail_on:
            raise ModelError("boom")
        return {"titulo": f"unidad de {len(images)}",
                "se_entiende_sola": self.complete_at.get(len(images), True)}


def screens(n: int) -> list[Screen]:
    return [Screen(index=i, start=float(i * 10), end=float((i + 1) * 10),
                   frame=Path(f"s{i}.png")) for i in range(n)]


def transcript() -> Transcript:
    return Transcript("es", [Utterance(0.0, 60.0, "habla")])


def test_complete_units_are_never_merged():
    model = FakeModel({1: True})
    units = analyze(model, screens(4), transcript())
    assert len(units) == 4
    assert model.calls == [1, 1, 1, 1]
    assert not any(u.merged for u in units)


def test_truncated_unit_absorbs_the_next_screen():
    """Una idea partida se reanaliza con las dos pantallas juntas."""
    model = FakeModel({1: False, 2: True})
    units = analyze(model, screens(4), transcript())
    assert [len(u.screens) for u in units] == [2, 2]
    assert model.calls == [1, 2, 1, 2]  # el intento de 1 se descarta al fusionar
    assert all(u.merged for u in units)


def test_merging_stops_at_the_cap():
    """Sin tope, una cadena de tramos truncados fusionaría el video entero."""
    model = FakeModel({1: False, 2: False, 3: False})
    units = analyze(model, screens(6), transcript())
    assert all(len(u.screens) <= MAX_SCREENS_PER_UNIT for u in units)
    assert all(u.link_pending for u in units)


def test_unit_at_the_cap_degrades_instead_of_failing():
    model = FakeModel({1: False, 2: False, 3: False})
    units = analyze(model, screens(3), transcript())
    assert len(units) == 1
    assert units[0].link_pending is True
    assert units[0].error is None


def test_last_screen_cannot_merge_forward():
    """No hay pantalla siguiente que absorber: se marca el enlace y se cierra."""
    model = FakeModel({1: False})
    units = analyze(model, screens(1), transcript())
    assert len(units) == 1
    assert units[0].link_pending is True


def test_every_screen_lands_in_exactly_one_unit():
    model = FakeModel({1: False, 2: True})
    units = analyze(model, screens(5), transcript())
    covered = [s.index for u in units for s in u.screens]
    assert covered == list(range(5))


def test_a_failed_screen_does_not_stop_the_run():
    model = FakeModel({1: True}, fail_on={1})
    units = analyze(model, screens(3), transcript())
    assert len(units) == 3
    assert all(u.error for u in units)


def test_progress_callbacks_report_merges():
    model = FakeModel({1: False, 2: True})
    spans: list[int] = []
    analyze(model, screens(2), transcript(), on_call=lambda i, span: spans.append(span))
    assert spans == [1, 2]
