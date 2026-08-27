"""Recorte del JSON y del bloque de razonamiento en la salida del modelo."""
import pytest

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


@pytest.mark.parametrize("text", ["sin json aquí", "", '{"a": 1'])
def test_unusable_output_raises(text):
    with pytest.raises(ValueError):
        _extract_json(text)


def test_reasoning_block_is_dropped():
    assert _strip_reasoning("<think>\ndivagando\n</think>\n\nrespuesta") == "respuesta"


def test_only_the_last_think_marker_counts():
    assert _strip_reasoning("<think>a</think>b</think>final") == "final"


def test_output_without_reasoning_is_untouched():
    assert _strip_reasoning("  respuesta directa  ") == "respuesta directa"
