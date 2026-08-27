"""Transcripción del audio y alineación de las ventanas con el habla.

Los cortes del video son visuales, pero las ideas son habladas: una explicación
cruza a menudo un cambio de pantalla ("ahora configuramos el…" [cambia] "…usuario
del módulo"). Cortar la ventana de audio justo en el cambio visual mutila la
frase.

Whisper ya segmenta por pausas naturales, así que sus fronteras son un buen
proxy de "fin de idea". `snap_to_speech` mueve cada corte visual a la frontera
de segmento más cercana, lo que elimina los cortes a mitad de oración sin
costar una sola llamada a un modelo.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}

#: Cuánto puede moverse un corte visual para alinearse al habla. Más allá de
#: esto la pantalla y su audio quedarían visiblemente desfasados, así que se
#: prefiere el corte visual crudo.
SNAP_TOLERANCE = 4.0


@dataclass(frozen=True)
class Utterance:
    """Un segmento de habla con sus tiempos."""

    start: float
    end: float
    text: str

    def stamped(self) -> str:
        return f"[{_ts(self.start)} → {_ts(self.end)}] {self.text.strip()}"


@dataclass(frozen=True)
class Transcript:
    language: str
    utterances: list[Utterance]

    @property
    def full_text(self) -> str:
        return " ".join(u.text.strip() for u in self.utterances).strip()

    def between(self, start: float, end: float) -> list[Utterance]:
        """Segmentos cuyo centro cae dentro de [start, end).

        Se usa el centro y no el solape para que cada segmento pertenezca a
        exactamente una ventana: sin eso, una frase a caballo entre dos
        pantallas se contaría dos veces y el resumen final la repetiría.
        """
        return [u for u in self.utterances
                if start <= (u.start + u.end) / 2 < end]

    def stamped_between(self, start: float, end: float) -> str:
        return "\n".join(u.stamped() for u in self.between(start, end))


def has_audio(video: Path) -> bool:
    """¿El archivo trae alguna pista de audio?"""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True)
    return "audio" in proc.stdout


def transcribe(video: Path, model: str = "large-v3-turbo",
               language: str | None = None) -> Transcript:
    """Transcribe la pista de audio del video con MLX Whisper.

    Una grabación muda devuelve una transcripción vacía en vez de fallar: es un
    caso legítimo, y además es donde el análisis visual más aporta, porque toda
    la información está en las pantallas.
    """
    if not has_audio(video):
        return Transcript(language="—", utterances=[])

    import mlx_whisper  # import perezoso: carga MLX, que tarda

    kwargs: dict = {"path_or_hf_repo": MODELS[model], "verbose": False}
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(video), **kwargs)

    utterances = [
        Utterance(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in result.get("segments", [])
        if s.get("text", "").strip()
    ]
    return Transcript(language=result.get("language", "?"), utterances=utterances)


def snap_to_speech(cuts: list[float], transcript: Transcript,
                   tolerance: float = SNAP_TOLERANCE,
                   duration: float | None = None) -> list[float]:
    """Extiende hacia adelante los cortes que parten una frase por la mitad.

    Solo se mueve un corte cuando un segmento de habla lo cruza, y solo hasta
    el final de ese segmento. Las dos restricciones importan:

    - **Solo si algo lo cruza.** Un corte que cae en un silencio ya es una
      frontera limpia; arrastrarlo a la frontera "más cercana" solo lo aleja
      del momento en que la pantalla realmente cambió.
    - **Solo hacia adelante.** La frase se queda con la pantalla en la que
      empezó a decirse. Mover el corte hacia atrás haría lo contrario:
      atribuiría a la pantalla siguiente algo que se dijo mientras se veía la
      anterior, que es el error que esto viene a evitar.

    Si la frase que cruza es tan larga que alinearla desfasaría la ventana más
    de `tolerance`, se deja el corte visual: vale más una frase partida que una
    pantalla emparejada con el audio equivocado.

    `duration` acota el último corte. No es opcional por gusto: Whisper emite
    timestamps que se pasan del final del medio — medido, un segmento que
    termina en 318.8s sobre un video de 305.9s — y sin ese tope el último corte
    se va fuera del video, donde no hay ningún frame que extraer.
    """
    if not transcript.utterances or not cuts:
        return list(cuts)

    snapped = [cuts[0]]
    for i, cut in enumerate(cuts[1:], start=1):
        # Alinear nunca puede saltar por encima del corte siguiente: dos
        # cambios de pantalla seguidos con una frase encima producirían una
        # ventana de duración negativa. Para el último corte el techo es el
        # final del video.
        if i + 1 < len(cuts):
            ceiling = cuts[i + 1]
        else:
            ceiling = duration if duration is not None else float("inf")

        candidate = cut
        for utterance in transcript.utterances:
            if utterance.start < cut < utterance.end:
                if utterance.end - cut <= tolerance and utterance.end < ceiling:
                    candidate = utterance.end
                break
        snapped.append(candidate if candidate > snapped[-1] else cut)
    return snapped


def _ts(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"
