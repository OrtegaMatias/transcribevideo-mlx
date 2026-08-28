"""Recorte del JSON y del bloque de razonamiento en la salida del modelo."""
import pytest

import transcribevideo_mlx.vlm as vlm_mod
from transcribevideo_mlx.vlm import _extract_json, _strip_reasoning


def test_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_object_wrapped_in_a_code_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_object_preceded_by_prose():
    """Los modelos anteponen una frase por más que se les prohíba."""
    assert _extract_json('Claro, aquí tienes:\n{"a": 1}\nEspero que sirva.') == {"a": 1}


def test_nested_braces():
    assert _extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}


def test_braces_inside_strings_do_not_end_the_object():
    parsed = _extract_json('{"texto_en_pantalla": "usa {llaves} en la UI", "n": 2}')
    assert parsed["n"] == 2


def test_escaped_quote_inside_a_string():
    parsed = _extract_json(r'{"texto_en_pantalla": "dice \"Guardar\"", "n": 3}')
    assert parsed["n"] == 3


def test_a_backslash_read_off_the_screen_does_not_lose_the_unit():
    """El modelo copia lo que ve, y una barra invertida llega tal cual.

    Visto en un video real: `Kids\\Proteger`. Para JSON es un escape inválido y
    `json.loads` rechaza el objeto entero, así que se perdía la pantalla
    completa por un carácter. Reintentar no sirve: a temperatura cero el modelo
    devuelve lo mismo.
    """
    raw = '{"titulo": "Botones", "texto_en_pantalla": "Kids\\Proteger la batería"}'
    assert _extract_json(raw)["texto_en_pantalla"] == "Kids\\Proteger la batería"


def test_valid_escapes_survive_the_repair():
    parsed = _extract_json(r'{"a": "linea1\nlinea2", "b": "dice \"hola\"", "c": "c:\\ruta"}')
    assert parsed["a"] == "linea1\nlinea2"
    assert parsed["b"] == 'dice "hola"'
    assert parsed["c"] == "c:\\ruta"


def test_unicode_escape_is_not_broken_by_the_repair():
    assert _extract_json(r'{"a": "\u00f1andu"}')["a"] == "ñandu"


@pytest.mark.parametrize("text", ["sin json aquí", "", '{"a": 1'])
def test_unusable_output_raises(text):
    with pytest.raises(ValueError):
        _extract_json(text)


def test_qwen_reasoning_block_is_dropped():
    assert _strip_reasoning("<think>\ndivagando\n</think>\n\nrespuesta") == "respuesta"


def test_gemma_reasoning_channel_is_dropped():
    """Cada familia cierra el canal a su manera.

    gemma abre `<|channel>thought` y cierra con `<channel|>`, con la barra al
    otro lado. Conocer solo el marcador de Qwen dejaba pasar el razonamiento
    crudo — y en inglés — al informe final.
    """
    raw = "<|channel>thought\nplanning the answer\n<channel|>\n\n## Resumen\nok"
    assert _strip_reasoning(raw) == "## Resumen\nok"


def test_only_the_last_marker_counts():
    assert _strip_reasoning("<think>a</think>b</think>final") == "final"


def test_the_latest_closer_wins_across_families():
    assert _strip_reasoning("</think>medio<channel|>final") == "final"


def test_output_without_reasoning_is_untouched():
    assert _strip_reasoning("  respuesta directa  ") == "respuesta directa"


# ── resguardo de memoria ────────────────────────────────────────────────

def test_a_healthy_machine_is_not_blocked(monkeypatch):
    """Medir la memoria LIBRE del momento resultó demasiado nervioso.

    Con 20 GB libres de 48, el resguardo original rechazaba una corrida que
    cabía perfectamente: macOS comprime y pagina, así que la libre instantánea
    no dice si algo cabe. Se compara contra la física.
    """
    monkeypatch.setattr(vlm_mod, "total_memory_gb", lambda: 48.0)
    monkeypatch.setattr(vlm_mod, "other_instances", lambda: 0)
    assert vlm_mod.check_headroom("org/gemma", 16.0) is None


def test_a_machine_that_truly_cannot_fit_it_is_blocked(monkeypatch):
    monkeypatch.setattr(vlm_mod, "total_memory_gb", lambda: 16.0)
    monkeypatch.setattr(vlm_mod, "other_instances", lambda: 0)
    aviso = vlm_mod.check_headroom("org/gemma-4-26B", 16.0)
    assert aviso and "16 GB en total" in aviso


def test_a_second_run_is_what_actually_kills_the_machine(monkeypatch):
    """Ocurrió de verdad: dos corridas de 21 GB en una máquina de 48.

    El sistema no dio error, se congeló y hubo que reiniciarlo a la fuerza. La
    memoria libre no lo anticipa porque cada proceso la toma de a poco; contar
    procesos sí.
    """
    monkeypatch.setattr(vlm_mod, "total_memory_gb", lambda: 48.0)
    monkeypatch.setattr(vlm_mod, "other_instances", lambda: 1)
    aviso = vlm_mod.check_headroom("org/gemma", 16.0)
    assert aviso and "ya hay 1 corrida" in aviso.lower()


def test_a_model_of_unknown_size_is_not_blocked(monkeypatch):
    """Si no se sabe cuánto pesa, no se puede afirmar que no cabe."""
    monkeypatch.setattr(vlm_mod, "total_memory_gb", lambda: 8.0)
    assert vlm_mod.check_headroom("org/modelo", 0.0) is None


def test_the_machine_reports_its_real_size():
    assert vlm_mod.total_memory_gb() > 1
    assert vlm_mod.available_memory_gb() > 0


def test_this_very_process_is_not_counted_as_a_rival():
    """Contarse a sí mismo bloquearía toda corrida, siempre."""
    assert vlm_mod.other_instances() >= 0
