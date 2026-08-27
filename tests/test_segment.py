"""Detección de cambios de pantalla por dhash."""
import random

from transcribevideo_mlx.segment import (HASH_BITS, HASH_H, HASH_W, _dhash,
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
