"""Lectura literal con Vision, el OCR nativo de macOS."""
import pytest

from transcribevideo_mlx import ocr


def test_vision_is_available_on_this_machine():
    """Sin PyObjC el modo --ocr vision no puede funcionar."""
    assert ocr.available() is True


def test_reads_the_text_of_a_generated_screen(tmp_path):
    """Verdad de referencia conocida: la imagen la genera este test."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 320), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    for i, line in enumerate(["Gestion de Usuarios", "RUT 15.482.930-1",
                              "Perfil Administrador"]):
        draw.text((40, 40 + i * 70), line, font=font, fill=(20, 20, 20))
    path = tmp_path / "pantalla.png"
    img.save(path)

    leido = " ".join(ocr.read_text(path))
    assert "Gestion de Usuarios" in leido
    assert "15.482.930-1" in leido
    assert "Administrador" in leido


def test_a_blank_screen_yields_nothing(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    path = tmp_path / "vacia.png"
    Image.new("RGB", (400, 200), (240, 240, 240)).save(path)
    assert ocr.read_text(path) == []


def test_a_missing_file_does_not_explode(tmp_path):
    assert ocr.read_text(tmp_path / "no-existe.png") == []


def test_several_screens_are_labelled(tmp_path):
    """Al fusionar pantallas, el modelo debe saber cuál es cuál."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    paths = []
    for i, word in enumerate(["Primera", "Segunda"]):
        img = Image.new("RGB", (500, 150), (255, 255, 255))
        ImageDraw.Draw(img).text((30, 50), word, font=font, fill=(0, 0, 0))
        p = tmp_path / f"s{i}.png"
        img.save(p)
        paths.append(p)

    texto = ocr.read_screens(paths)
    assert "--- pantalla 1 ---" in texto and "--- pantalla 2 ---" in texto
    assert "Primera" in texto and "Segunda" in texto
