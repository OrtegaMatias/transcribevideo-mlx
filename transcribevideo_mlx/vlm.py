"""Cliente del modelo de visión, in-process sobre MLX.

Se carga el modelo directamente con `mlx-vlm` en vez de hablar con un servidor
local (LM Studio, Ollama). Además de evitar la dependencia de una app abierta,
la razón de fondo es el conteo de tokens: LM Studio no incluye los tokens
visuales en el `usage` que reporta — mide 75 tokens para un prompt que en
realidad ocupa 914 — así que no hay forma de presupuestar el contexto y la
corrida muere con un HTTP 400 sin aviso. En proceso, el número es real.

El razonamiento del modelo se administra por etapa: apagado para la extracción
por pantalla, que es una tarea de lectura y se acelera ~2.6x sin él, y
encendido para el informe final, que sí es una tarea de síntesis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MODEL = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
#: Techo de generación por pantalla. Un chunk típico ronda los 400 tokens; el
#: margen cubre pantallas con mucho texto sin dejar que una repetición se
#: descontrole.
CHUNK_MAX_TOKENS = 1600
REPORT_MAX_TOKENS = 4000


class ModelError(RuntimeError):
    """El modelo no se pudo cargar o devolvió algo inutilizable."""


def resolve_model_path(model: str = DEFAULT_MODEL) -> str:
    """Prefiere una copia local ya descargada por LM Studio.

    Evita bajar 16 GB de nuevo a quien ya tiene el modelo en su Mac. Si no está,
    se devuelve el id de Hugging Face y `mlx-vlm` lo descarga.
    """
    local = Path.home() / ".lmstudio" / "models" / model
    if (local / "config.json").exists():
        return str(local)
    return model


class VisionModel:
    """Envuelve el modelo cargado. Se instancia una vez por corrida."""

    def __init__(self, model: str = DEFAULT_MODEL):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        path = resolve_model_path(model)
        try:
            self._model, self._processor = load(path)
            self._config = load_config(path)
        except Exception as exc:  # noqa: BLE001 - la causa se muestra al usuario
            raise ModelError(
                f"No se pudo cargar el modelo '{model}'.\n{exc}") from exc
        self.name = model

    def analyze_screen(self, system: str, user: str,
                       images: list[Path]) -> dict:
        """Analiza un tramo y devuelve el chunk ya parseado.

        Un modelo que devuelve JSON inválido casi siempre lo arregla al
        reintentar; si falla dos veces se propaga el error y el tramo se marca
        como fallido en vez de tumbar la corrida entera.
        """
        raw = self._generate(system, user, images,
                             max_tokens=CHUNK_MAX_TOKENS, thinking=False)
        try:
            return _extract_json(raw)
        except ValueError:
            retry = self._generate(
                system,
                user + "\n\nDevuelve SOLO el objeto JSON, sin ningún otro texto.",
                images, max_tokens=CHUNK_MAX_TOKENS, thinking=False)
            try:
                return _extract_json(retry)
            except ValueError as exc:
                raise ModelError(
                    f"El modelo no devolvió JSON válido: {retry[:200]}") from exc

    def write_report(self, system: str, user: str) -> str:
        """Redacta el informe final. Sin imágenes: solo texto ya extraído."""
        raw = self._generate(system, user, [],
                             max_tokens=REPORT_MAX_TOKENS, thinking=True,
                             reasoning_effort="medium")
        return raw.strip()

    def _generate(self, system: str, user: str, images: list[Path],
                  max_tokens: int, thinking: bool,
                  reasoning_effort: str = "low") -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        template_kwargs: dict = {"enable_thinking": thinking}
        if thinking:
            template_kwargs["reasoning_effort"] = reasoning_effort

        prompt = apply_chat_template(
            self._processor, self._config, messages,
            num_images=len(images), **template_kwargs)

        result = generate(
            self._model, self._processor, prompt,
            image=[str(p) for p in images] or None,
            max_tokens=max_tokens, temperature=0.0, verbose=False)

        text = result.text if hasattr(result, "text") else str(result)
        return _strip_reasoning(text)


def _strip_reasoning(text: str) -> str:
    """Descarta el bloque de razonamiento.

    Con `enable_thinking=False` la plantilla ya deja `<think></think>` cerrado y
    la salida viene limpia; con razonamiento activo el texto útil empieza
    después del `</think>` final.
    """
    marker = "</think>"
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    return text.strip()


def _extract_json(text: str) -> dict:
    """Recorta el primer objeto JSON balanceado del texto.

    Los modelos envuelven el JSON en vallas de código o lo preceden de una
    frase por más que se les prohíba, así que se busca el objeto en vez de
    confiar en que la respuesta entera sea válida.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no hay objeto JSON en la respuesta")

    depth, in_string, escaped = 0, False, False
    for i, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("objeto JSON incompleto")
