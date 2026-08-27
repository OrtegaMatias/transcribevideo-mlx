"""Alineación de cortes visuales con las fronteras del habla."""
import pytest

from transcribevideo_mlx.audio import Transcript, Utterance, snap_to_speech


def transcript(*spans: tuple[float, float]) -> Transcript:
    return Transcript("es", [Utterance(a, b, "texto") for a, b in spans])


def test_cut_in_silence_is_left_alone():
    """Un corte que no parte ninguna frase ya es una frontera limpia.

    Moverlo a la frontera de habla "más cercana" solo lo alejaría del instante
    en que la pantalla realmente cambió.
    """
    assert snap_to_speech([0.0, 10.0], transcript((1, 5), (15, 20))) == [0.0, 10.0]


def test_cut_splitting_a_sentence_extends_forward():
    assert snap_to_speech([0.0, 10.0], transcript((9, 13.9))) == [0.0, 13.9]


def test_snap_never_moves_backward():
    """La frase se queda con la pantalla en la que empezó a decirse."""
    snapped = snap_to_speech([0.0, 10.0], transcript((9, 13.9)))
    assert snapped[1] >= 10.0


def test_long_sentence_beyond_tolerance_keeps_the_visual_cut():
    """Vale más una frase partida que una pantalla con el audio equivocado."""
    assert snap_to_speech([0.0, 10.0], transcript((9, 30))) == [0.0, 10.0]


def test_snap_never_crosses_the_next_cut():
    """Dos cambios seguidos con una frase encima no pueden invertir el orden."""
    snapped = snap_to_speech([0.0, 10.0, 11.0], transcript((9, 12.5)))
    assert snapped == sorted(snapped)
    assert len(set(snapped)) == len(snapped)


@pytest.mark.parametrize("cuts", [[], [0.0], [0.0, 5.0, 30.0, 61.0]])
def test_output_is_always_strictly_increasing(cuts):
    tr = transcript((1, 4.4), (4.7, 8.1), (18.2, 20.3), (28, 32.5), (32, 40.7))
    snapped = snap_to_speech(cuts, tr)
    assert len(snapped) == len(cuts)
    assert all(a < b for a, b in zip(snapped, snapped[1:]))


def test_no_speech_leaves_cuts_untouched():
    assert snap_to_speech([0.0, 10.0], transcript()) == [0.0, 10.0]


def test_between_assigns_each_utterance_to_exactly_one_window():
    """Una frase a caballo entre dos pantallas no puede contarse dos veces."""
    tr = transcript((0, 4), (4, 9), (9, 14), (14, 20))
    windows = [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]
    assigned = [u for start, end in windows for u in tr.between(start, end)]
    assert len(assigned) == len(tr.utterances)
    assert len(set(id(u) for u in assigned)) == len(tr.utterances)
