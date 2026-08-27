"""Segmentación de un screencast en las pantallas únicas que lo componen.

Deliberadamente NO se usa el filtro `scene` de ffmpeg. Ese filtro mide
diferencia global de píxeles y está calibrado para cortes de película; en una
grabación de pantalla dos vistas del mismo sistema comparten fondo, cabecera y
layout, así que un cambio real de pantalla puntúa ~0.05 — muy por debajo del
0.3-0.4 habitual, y mezclado con el ruido de los keyframes del encoder.

En su lugar se muestrea el video a baja resolución en escala de grises y se
comparan dhash de 1024 bits. Medido sobre pantallas de UI, ese hash separa
"pantallas distintas" (92-148 bits de distancia) de "la misma pantalla"
(4 bits) por un factor ~23x, con una meseta estable de umbrales entre 20 y 60.
La misma pasada resuelve la detección de cortes y el dedup, y cuesta ~3 s para
media hora de video.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# dhash de 32x32 necesita una columna extra para comparar pares adyacentes.
HASH_W, HASH_H = 33, 32
HASH_BITS = (HASH_W - 1) * HASH_H  # 1024

SAMPLE_FPS = 2.0
#: Distancia de Hamming (sobre 1024) a partir de la cual dos frames son
#: pantallas distintas. Ver el módulo docstring para la calibración.
CUT_THRESHOLD = 50
#: Segundos a esperar tras un corte antes de capturar el frame representativo,
#: para no fotografiar una animación de transición a medio camino.
SETTLE_SECONDS = 1.0
#: Ancho máximo del frame que se le manda al VLM. A 1280px una pantalla cuesta
#: ~900 tokens visuales en Qwen3-VL; subirlo encarece cada llamada sin mejorar
#: el OCR de una UI.
FRAME_MAX_WIDTH = 1280


class FFmpegError(RuntimeError):
    """ffmpeg no está disponible o falló procesando el video."""


@dataclass(frozen=True)
class Screen:
    """Una pantalla única del video y el tramo en que estuvo visible."""

    index: int
    start: float
    end: float
    frame: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


def probe_duration(video: Path) -> float:
    """Duración del video en segundos."""
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ])
    try:
        return float(out.strip())
    except ValueError as exc:
        raise FFmpegError(f"No se pudo leer la duración de {video.name}") from exc


def sample_hashes(video: Path, fps: float = SAMPLE_FPS) -> list[int]:
    """Muestrea el video y devuelve un dhash por frame muestreado.

    Decodifica directamente a 33x32 en gris y lee el raw por stdout: para media
    hora de video son ~3.600 frames de 1 KB, así que nunca toca el disco.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={HASH_W}:{HASH_H}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode(errors="replace").strip()[:500])

    raw, size = proc.stdout, HASH_W * HASH_H
    return [_dhash(raw[i * size:(i + 1) * size])
            for i in range(len(raw) // size)]


def find_cuts(hashes: list[int], fps: float = SAMPLE_FPS,
              threshold: int = CUT_THRESHOLD) -> list[float]:
    """Instantes (en segundos) en que la pantalla cambia.

    Cada frame se compara contra el representante del tramo actual, no contra
    el frame anterior: así un cambio gradual (un fundido, una lista que se va
    poblando) se detecta una sola vez, cuando ya se alejó lo suficiente del
    punto de partida, en vez de no detectarse nunca por avanzar de a poco.
    """
    if not hashes:
        return []
    cuts, reference = [0.0], hashes[0]
    for i, current in enumerate(hashes[1:], start=1):
        if (reference ^ current).bit_count() > threshold:
            cuts.append(i / fps)
            reference = current
    return cuts


def extract_screens(video: Path, cuts: list[float], duration: float,
                    out_dir: Path) -> list[Screen]:
    """Extrae en resolución completa un frame representativo por tramo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = cuts + [duration]
    screens = []
    for i, start in enumerate(cuts):
        end = bounds[i + 1]
        at = _representative_instant(start, end)
        frame = out_dir / f"screen-{i:04d}.png"
        _grab_frame(video, at, frame)
        screens.append(Screen(index=i, start=start, end=end, frame=frame))
    return screens


def _representative_instant(start: float, end: float) -> float:
    """Instante a capturar dentro de un tramo.

    Se espera un momento tras el corte para no capturar una transición, pero
    nunca más allá de la mitad de un tramo corto.
    """
    return start + min(SETTLE_SECONDS, max(0.0, (end - start) / 2))


def _grab_frame(video: Path, at: float, dest: Path) -> None:
    # -ss antes de -i hace seek por keyframe: impreciso al milisegundo pero
    # instantáneo, y la pantalla es estable dentro de su tramo.
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", str(video),
         "-frames:v", "1",
         "-vf", f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
         str(dest)],
        capture_output=True,
    )
    if proc.returncode != 0 or not dest.exists():
        raise FFmpegError(
            f"No se pudo extraer el frame en {at:.1f}s: "
            f"{proc.stderr.decode(errors='replace').strip()[:300]}")


def _dhash(buf: bytes) -> int:
    """dhash de 1024 bits: cada píxel comparado con su vecino de la derecha."""
    bits = 0
    for row in range(HASH_H):
        offset = row * HASH_W
        for col in range(HASH_W - 1):
            bits = (bits << 1) | (buf[offset + col] > buf[offset + col + 1])
    return bits


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.strip()[:500])
    return proc.stdout
