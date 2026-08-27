"""Detección de cambios de pantalla por dhash."""
import random

from transcribevideo_mlx.segment import (HASH_BITS, HASH_H, HASH_W,
                                         absorb_transients, _dhash,
                                         _representative_instant, find_cuts)

SIZE = HASH_W * HASH_H


def noise_frame(seed: int) -> bytes:
    """Frame con estructura local, que es lo que dhash mide.

    Una rampa lineal no sirve como fixture: dhash compara cada píxel con su
    vecino de la derecha, y en una rampa esa comparación da lo mismo para
    cualquier desplazamiento, así que dos frames "distintos" tendrían hashes
    idénticos.
    """
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(SIZE))


def drifted(base: bytes, pixels: int, seed: int = 99) -> bytes:
    """El mismo frame con `pixels` píxeles invertidos."""
    rng = random.Random(seed)
    buf = bytearray(base)
    positions = list(range(SIZE))
    rng.shuffle(positions)
    for i in positions[:pixels]:
        buf[i] = 255 - buf[i]
    return bytes(buf)


def test_hash_width_is_1024_bits():
    assert HASH_BITS == 1024
    assert (_dhash(noise_frame(1)) ^ _dhash(noise_frame(2))).bit_count() <= HASH_BITS


def test_identical_frames_have_distance_zero():
    assert (_dhash(noise_frame(7)) ^ _dhash(noise_frame(7))).bit_count() == 0


def test_different_frames_are_far_apart():
    distance = (_dhash(noise_frame(1)) ^ _dhash(noise_frame(2))).bit_count()
    assert distance > 300


def test_flat_frame_hashes_to_zero():
    """Un frame sin estructura horizontal no tiene bits encendidos."""
    assert _dhash(bytes([128] * SIZE)) == 0


def test_no_cuts_when_nothing_changes():
    hashes = [_dhash(noise_frame(4))] * 20
    assert find_cuts(hashes, fps=2.0) == [0.0]


def test_cut_detected_at_the_right_second():
    hashes = [_dhash(noise_frame(1))] * 10 + [_dhash(noise_frame(2))] * 10
    assert find_cuts(hashes, fps=2.0) == [0.0, 5.0]


def test_small_change_stays_below_threshold():
    """Un cursor que se mueve no puede contar como pantalla nueva."""
    base = noise_frame(1)
    hashes = [_dhash(base)] * 5 + [_dhash(drifted(base, 6))] * 5
    assert find_cuts(hashes, fps=2.0) == [0.0]


def test_comparison_is_against_the_run_representative():
    """Una deriva lenta se detecta cuando ya se alejó del punto de partida.

    Comparando contra el frame anterior, cada paso quedaría bajo el umbral y el
    cambio pasaría inadvertido para siempre por más que la pantalla terminara
    siendo otra.
    """
    base = noise_frame(1)
    hashes = [_dhash(drifted(base, step * 40)) for step in range(12)]

    consecutive = max(
        (hashes[i] ^ hashes[i + 1]).bit_count() for i in range(len(hashes) - 1))
    assert consecutive < 50, "cada paso individual debe ser pequeño"
    assert len(find_cuts(hashes, fps=1.0)) >= 2


def test_empty_input():
    assert find_cuts([], fps=2.0) == []


def test_representative_instant_waits_out_the_transition():
    assert _representative_instant(10.0, 40.0) == 11.0


def test_representative_instant_never_passes_a_short_screen():
    """En un tramo corto se captura en la mitad, no fuera de él."""
    assert _representative_instant(10.0, 11.0) == 10.5


def test_a_burst_of_changes_collapses_to_where_it_settled():
    """De una animación sobrevive el instante en que la pantalla se estabiliza.

    Así el frame representativo sale de contenido asentado y no de un fotograma
    a mitad de la transición.
    """
    cuts = [0.0, 10.0, 10.4, 10.8, 11.2, 30.0]
    assert absorb_transients(cuts, 45.0, 2.0) == [0.0, 11.2, 30.0]


def test_screens_that_persist_are_kept():
    cuts = [0.0, 10.0, 14.0, 30.0]
    assert absorb_transients(cuts, 45.0, 2.0) == cuts


def test_the_start_of_the_video_is_always_a_screen():
    """Aunque el primer tramo sea breve, 0.0 es el inicio y no se puede fundir."""
    assert absorb_transients([0.0, 0.5, 20.0], 40.0, 2.0)[0] == 0.0


def test_the_last_screen_is_measured_against_the_video_end():
    assert absorb_transients([0.0, 30.0], 31.0, 2.0) == [0.0]
    assert absorb_transients([0.0, 30.0], 40.0, 2.0) == [0.0, 30.0]


def test_absorbing_never_loses_time():
    """El video entero sigue cubierto: los tramos fundidos no dejan huecos."""
    cuts = [0.0, 5.0, 5.3, 5.6, 20.0, 20.2, 40.0]
    kept = absorb_transients(cuts, 50.0, 2.0)
    bounds = kept + [50.0]
    assert kept[0] == 0.0
    assert all(b > a for a, b in zip(bounds, bounds[1:]))
    assert bounds[-1] == 50.0


def test_empty_cuts():
    assert absorb_transients([], 10.0, 2.0) == []
