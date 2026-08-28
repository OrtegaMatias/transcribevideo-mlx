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

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

#: Modelo de la etapa de lectura, la que se repite una vez por pantalla. Se
#: elige por velocidad: medido sobre las mismas pantallas, el MoE de gemma
#: (128 expertos, 8 activos) resuelve una pantalla en 3.2s contra 14.7s del
#: Qwen denso, con fidelidad idéntica en el cuerpo de la pantalla.
DEFAULT_MODEL = "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
#: Modelo del informe final. Es UNA sola llamada, y es la única etapa de
#: síntesis de la corrida, así que aquí conviene el modelo más capaz aunque sea
#: más lento: pagar dos minutos una vez, en vez de once segundos cuarenta veces.
DEFAULT_REPORTER = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
#: Techo de generación por pantalla. Un chunk típico ronda los 400 tokens; el
#: margen cubre pantallas con mucho texto sin dejar que una repetición se
#: descontrole.
CHUNK_MAX_TOKENS = 1600
REPORT_MAX_TOKENS = 4000


class ModelError(RuntimeError):
    """El modelo no se pudo cargar o devolvió algo inutilizable."""


@dataclass
class Usage:
    """Lo que costó una llamada. Se acumula a lo largo de la corrida."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0
    peak_memory: float = 0.0
    #: Tokens por segundo de la última llamada.
    tps: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.generation_tokens += other.generation_tokens
        self.calls += other.calls
        self.seconds += other.seconds
        self.peak_memory = max(self.peak_memory, other.peak_memory)
        self.tps = other.tps


@dataclass
class Generation:
    text: str = ""
    usage: Usage = field(default_factory=Usage)


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

    def release(self) -> None:
        """Suelta los pesos para que quepa otro modelo.

        Los dos modelos suman ~32 GB y cada uno tiene un pico medido de 18-21 GB,
        así que no caben a la vez en 48 GB de memoria unificada. Cargarlos en
        secuencia cuesta unos segundos y evita quedarse sin memoria justo al
        final, con todo el trabajo caro ya hecho.
        """
        self._model = self._processor = self._config = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:  # noqa: BLE001 - liberar es best-effort
            pass

    def analyze_screen(self, system: str, user: str, images: list[Path],
                       on_delta: Callable[[str], None] | None = None
                       ) -> tuple[dict, Usage]:
        """Analiza un tramo y devuelve el chunk parseado y lo que costó.

        Un modelo que devuelve JSON inválido casi siempre lo arregla al
        reintentar; si falla dos veces se propaga el error y el tramo se marca
        como fallido en vez de tumbar la corrida entera.
        """
        run = self._generate(system, user, images, max_tokens=CHUNK_MAX_TOKENS,
                             thinking=False, on_delta=on_delta)
        try:
            return _extract_json(run.text), run.usage
        except ValueError:
            retry = self._generate(
                system,
                user + "\n\nDevuelve SOLO el objeto JSON, sin ningún otro texto.",
                images, max_tokens=CHUNK_MAX_TOKENS, thinking=False,
                on_delta=on_delta)
            run.usage.add(retry.usage)
            try:
                return _extract_json(retry.text), run.usage
            except ValueError as exc:
                raise ModelError(
                    f"El modelo no devolvió JSON válido: {retry.text[:200]}") from exc

    def write_report(self, system: str, user: str,
                     on_delta: Callable[[str], None] | None = None
                     ) -> tuple[str, Usage]:
        """Redacta el informe final. Sin imágenes: solo texto ya extraído."""
        run = self._generate(system, user, [], max_tokens=REPORT_MAX_TOKENS,
                             thinking=True, reasoning_effort="medium",
                             on_delta=on_delta)
        return run.text.strip(), run.usage

    def _generate(self, system: str, user: str, images: list[Path],
                  max_tokens: int, thinking: bool,
                  reasoning_effort: str = "low",
                  on_delta: Callable[[str], None] | None = None) -> Generation:
        import time

        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        template_kwargs: dict = {"enable_thinking": thinking}
        if thinking:
            template_kwargs["reasoning_effort"] = reasoning_effort

        prompt = apply_chat_template(
            self._processor, self._config, messages,
            num_images=len(images), **template_kwargs)

        started = time.time()
        pieces: list[str] = []
        usage = Usage(calls=1)

        for chunk in stream_generate(
                self._model, self._processor, prompt,
                image=[str(p) for p in images] or None,
                max_tokens=max_tokens, temperature=0.0):
            if chunk.text:
                pieces.append(chunk.text)
                if on_delta:
                    on_delta(chunk.text)
            usage.prompt_tokens = chunk.prompt_tokens or usage.prompt_tokens
            usage.generation_tokens = (chunk.generation_tokens
                                       or usage.generation_tokens)
            usage.tps = chunk.generation_tps or usage.tps
            usage.peak_memory = chunk.peak_memory or usage.peak_memory

        usage.seconds = time.time() - started
        return Generation(text=_strip_reasoning("".join(pieces)), usage=usage)


#: Cada familia cierra su canal de razonamiento a su manera. Qwen usa
#: `</think>`; gemma abre `<|channel>thought` y cierra con `<channel|>`, con la
#: barra al otro lado. Conocer solo una deja que el razonamiento crudo — y en su
#: idioma de entrenamiento — se cuele entero al informe.
REASONING_CLOSERS = ("</think>", "<channel|>")


def _strip_reasoning(text: str) -> str:
    """Descarta el bloque de razonamiento, sea cual sea el modelo.

    Con `enable_thinking=False` la plantilla deja el canal ya cerrado y la
    salida viene limpia; con razonamiento activo el texto útil empieza después
    del último cierre.
    """
    cut = max((text.rfind(marker) + len(marker) for marker in REASONING_CLOSERS
               if marker in text), default=0)
    return text[cut:].strip()


#: Escapes que JSON acepta detrás de una contrabarra.
_VALID_ESCAPES = set('"\\/bfnrtu')


def _repair_escapes(chunk: str) -> str:
    """Duplica las contrabarras que no inician un escape válido.

    Transcribiendo una interfaz el modelo copia lo que ve, y una barra invertida
    en pantalla llega tal cual: `Kids\\Proteger`. Para JSON eso es un escape
    inválido y `json.loads` rechaza el objeto entero — se pierde la pantalla
    completa por un carácter. Reintentar no ayuda: a temperatura cero el modelo
    devuelve exactamente lo mismo. Hay que repararlo.
    """
    out, i = [], 0
    while i < len(chunk):
        char = chunk[i]
        if char == "\\":
            following = chunk[i + 1] if i + 1 < len(chunk) else ""
            out.append("\\" if following in _VALID_ESCAPES else "\\\\")
        else:
            out.append(char)
        i += 1
    return "".join(out)


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
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return json.loads(_repair_escapes(candidate))
    raise ValueError("objeto JSON incompleto")


def available_memory_gb() -> float:
    """Memoria realmente disponible ahora mismo, en GB.

    Se suman las páginas libres y las inactivas: macOS puede reclamar las
    inactivas sin ir a disco, así que son memoria utilizable de verdad, no
    ocupada.
    """
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001 - sin vm_stat no se puede afirmar nada
        return float("inf")

    page = 16384
    pages = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
            pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page / 1e9


def total_memory_gb() -> float:
    """Memoria física de la máquina."""
    import subprocess
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout
        return int(out.strip()) / 1e9
    except Exception:  # noqa: BLE001
        return float("inf")


def other_instances() -> int:
    """Cuántas OTRAS corridas de esta herramienta hay vivas.

    Es el riesgo real, y el que se materializó: dos procesos pidiendo 21 GB cada
    uno en una máquina de 48 dejaron el sistema colgado. La memoria libre del
    momento no lo anticipa —cada proceso la toma de a poco— pero contar
    procesos sí.

    Lo delicado es no contarse a uno mismo. Basta con excluir el propio PID:
    el entorno virtual se llama `transcribevideo-mlx`, así que cualquier proceso
    hijo —el `resource_tracker` de multiprocessing, por ejemplo— lleva ese
    nombre en la ruta de su intérprete y aparece en la búsqueda. Y cuando la
    corrida la lanza la aplicación de escritorio, el padre también se llama
    igual. Por eso se compara el **grupo de procesos**: todo lo que esta corrida
    creó, y quien la creó, comparten grupo; otra corrida tiene el suyo.
    """
    import os
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", "transcribevideo"],
                             capture_output=True, text=True).stdout
        propio_grupo = os.getpgrp()
    except Exception:  # noqa: BLE001
        return 0

    ajenas = 0
    for token in out.split():
        if not token.isdigit():
            continue
        pid = int(token)
        try:
            if os.getpgid(pid) != propio_grupo:
                ajenas += 1
        except (ProcessLookupError, PermissionError):
            continue
    return ajenas


#: Memoria que se le deja al sistema operativo y al resto de aplicaciones.
OS_RESERVE_GB = 8.0


def check_headroom(model: str, needed_gb: float) -> str | None:
    """Avisa si esta corrida dejaría la máquina sin memoria.

    Se compara contra la memoria **física**, no contra la libre del momento.
    Medir la libre parecía lo correcto y resultó demasiado nervioso: macOS
    comprime y pagina, así que veinte gigas libres en una máquina de cuarenta y
    ocho no significan que no quepa, y rechazar por eso degrada corridas que
    habrían funcionado perfectamente.

    Lo que sí es un riesgo real, porque ocurrió: dos corridas simultáneas
    pidiendo veintiún gigas cada una dejaron el sistema colgado y hubo que
    reiniciar a la fuerza. Eso se detecta contando procesos, no bytes libres.
    """
    if not needed_gb:
        return None
    necesita = needed_gb * 1.35

    if (otras := other_instances()) > 0:
        return (f"Ya hay {otras} corrida(s) de transcribevideo en marcha. "
                f"Cada una usa cerca de {necesita:.0f} GB y dos a la vez no "
                "caben: el sistema no falla, se congela. "
                "Espera a que termine, o ciérrala.")

    total = total_memory_gb()
    if necesita > total - OS_RESERVE_GB:
        return (f"{model.split('/')[-1]} necesita cerca de {necesita:.0f} GB y "
                f"esta máquina tiene {total:.0f} GB en total. No alcanza: "
                "prueba con un modelo más pequeño usando --vlm.")
    return None
