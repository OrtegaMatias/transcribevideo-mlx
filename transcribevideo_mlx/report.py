"""Reduce final y serialización de resultados.

El informe se arma en dos capas. La redactada por el modelo va arriba; debajo
se añade una línea de tiempo generada de forma determinista a partir de las
unidades. Esa segunda capa importa: es la que permite auditar el informe contra
lo que realmente se leyó en pantalla, sin volver a procesar el video.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .audio import Transcript
from .pipeline import Unit
from .prompts import REPORT_SYSTEM, report_user_prompt
from .vlm import Usage, VisionModel


def write_report(model: VisionModel, units: list[Unit],
                 transcript: Transcript, on_delta=None) -> tuple[str, Usage]:
    """Pide al modelo el informe en Markdown a partir de las unidades."""
    return model.write_report(
        REPORT_SYSTEM,
        report_user_prompt(_chunks_block(units), transcript.full_text),
        on_delta)


def render_markdown(video: Path, body: str, units: list[Unit],
                    transcript: Transcript, duration: float,
                    elapsed: float) -> str:
    """Informe completo: la síntesis del modelo más el anexo auditable."""
    screens = sum(len(u.screens) for u in units)
    grouping = f"{screens} pantallas en {len(units)} unidades" if screens != len(units) \
        else f"{screens} pantallas"

    lines = [
        f"# {video.stem}",
        "",
        f"*{_ts(duration)} de video · {grouping} · "
        f"idioma {transcript.language} · "
        f"procesado en {_human(elapsed)} el {datetime.now():%Y-%m-%d %H:%M}*",
        "",
        body.strip(),
        "",
        "---",
        "",
        "## Línea de tiempo",
        "",
    ]

    for unit in units:
        marks = []
        if unit.merged:
            marks.append(f"{len(unit.screens)} pantallas fusionadas")
        if unit.link_pending:
            marks.append("continúa en la siguiente")
        suffix = f" — *{', '.join(marks)}*" if marks else ""

        lines.append(f"### {_ts(unit.start)} – {_ts(unit.end)} · {unit.title}{suffix}")
        lines.append("")

        if unit.error:
            lines += [f"> No se pudo analizar este tramo: {unit.error}", ""]
            continue

        if synthesis := _text(unit.chunk, "sintesis"):
            lines += [synthesis, ""]
        if screen_text := _text(unit.chunk, "texto_en_pantalla"):
            lines += ["**Texto en pantalla**", "", "```", screen_text, "```", ""]
        if elements := unit.chunk.get("elementos_ui") or []:
            lines += ["**Elementos**: " + ", ".join(str(e) for e in elements), ""]

    lines += ["---", "", "## Transcripción", "", transcript.full_text, ""]
    return "\n".join(lines)


def render_json(video: Path, units: list[Unit], transcript: Transcript,
                duration: float, elapsed: float, model_name: str,
                report_body: str, usage: Usage | None = None) -> str:
    """Todo lo intermedio, para regenerar el informe sin reprocesar el video."""
    payload = {
        "video": str(video),
        "duration": duration,
        "elapsed": elapsed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "language": transcript.language,
        "usage": asdict(usage) if usage else None,
        "report": report_body,
        "units": [
            {
                "index": unit.index,
                "start": unit.start,
                "end": unit.end,
                "merged_screens": len(unit.screens),
                "link_pending": unit.link_pending,
                "error": unit.error,
                "frames": [str(s.frame) for s in unit.screens],
                "usage": asdict(unit.usage),
                "chunk": unit.chunk,
            }
            for unit in units
        ],
        "transcript": [asdict(u) for u in transcript.utterances],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def unique_path(directory: Path, stem: str, ext: str) -> Path:
    """Nunca sobrescribe: agrega -1, -2… como hace transcribe-mlx."""
    candidate = directory / f"{stem}.{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}.{ext}"
        counter += 1
    return candidate


def _chunks_block(units: list[Unit]) -> str:
    blocks = []
    for unit in units:
        if unit.error:
            continue
        chunk = unit.chunk
        entry = [f"--- Pantalla {unit.index + 1} · {_ts(unit.start)} → {_ts(unit.end)}",
                 f"Título: {_text(chunk, 'titulo')}"]
        if text := _text(chunk, "texto_en_pantalla"):
            entry.append(f"Texto leído en pantalla:\n{text}")
        if elements := chunk.get("elementos_ui") or []:
            entry.append("Elementos: " + ", ".join(str(e) for e in elements))
        if narration := _text(chunk, "narracion"):
            entry.append(f"Narración: {narration}")
        if synthesis := _text(chunk, "sintesis"):
            entry.append(f"Síntesis: {synthesis}")
        if unit.link_pending:
            entry.append("NOTA: este tramo quedó truncado y continúa en el siguiente.")
        blocks.append("\n".join(entry))
    return "\n\n".join(blocks)


def _text(chunk: dict, key: str) -> str:
    """Campo de texto del chunk, tolerante a lo que devuelva el modelo.

    Un modelo que llena un campo con `null` o con un número no debe costar el
    informe entero después de quince minutos de proceso.
    """
    value = chunk.get(key)
    return str(value).strip() if value is not None else ""


def _ts(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _human(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
