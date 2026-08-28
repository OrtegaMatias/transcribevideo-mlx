"""Lectura literal de la pantalla con Vision, el OCR nativo de macOS.

Transcribir e interpretar son tareas de naturaleza opuesta, y conviene no
pedírselas al mismo modelo. Copiar los píxeles que son letras tiene una única
respuesta correcta y no admite juicio; decidir qué está pasando en la pantalla
es puro juicio. Un modelo de lenguaje es excelente en lo segundo y caro en lo
primero: genera token por token, y medido sobre 136 pantallas reales el campo de
texto literal se llevaba el 37% de toda la generación.

Peor que el costo es el sesgo. Un modelo que entiende puede *decidir* que algo
no vale la pena, y eso es una virtud al interpretar y un defecto al transcribir:
medido sobre una pantalla densa, el VLM recuperó 12 de 18 elementos conocidos y
se saltó justo los periféricos —la barra de estado, lo cortado por el borde—.
Vision no decide nada: reporta toda región de texto que encuentra.

Y hay un beneficio que no es de velocidad. Hoy el mismo modelo oye el audio y
escribe el texto de pantalla, así que puede colar en él algo que solo escuchó;
está mitigado por el prompt, pero el riesgo es estructural. Si el texto literal
lo produce algo que nunca oyó el audio, esa contaminación deja de ser posible.

A cambio, Vision es más ruidoso: intenta leer los iconos de la barra de estado
y produce cosas como "5t.l| 93%U". Por eso se filtra por confianza y el modelo
recibe este texto como material, no como verdad final.
"""
from __future__ import annotations

from pathlib import Path

#: Debajo de esta confianza, lo que Vision leyó suele ser un icono o un borde
#: interpretado como letras, no texto real.
MIN_CONFIDENCE = 0.3
#: Idiomas que se le indican al reconocedor. El orden importa: es la prioridad.
LANGUAGES = ["es-ES", "en-US"]


def available() -> bool:
    """¿Se puede usar Vision en esta máquina?"""
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - sin PyObjC, simplemente no está
        return False


def read_text(image: Path, min_confidence: float = MIN_CONFIDENCE) -> list[str]:
    """Devuelve las líneas de texto que Vision encuentra en la imagen.

    En el orden en que Vision las entrega, que sigue el flujo de lectura de la
    pantalla. Las regiones por debajo de `min_confidence` se descartan.
    """
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(image))
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        return []
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        return []

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request.setRecognitionLanguages_(LANGUAGES)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None)
    handler.performRequests_error_([request], None)

    lines: list[str] = []
    for observation in (request.results() or []):
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        best = candidates[0]
        if float(best.confidence()) < min_confidence:
            continue
        text = str(best.string()).strip()
        if _is_text(text):
            lines.append(text)
    return lines


def _is_text(line: str) -> bool:
    """¿Esto es texto de la interfaz, o un icono que el OCR intentó leer?

    Vision reporta toda región que se le parezca a escritura, y los iconos de
    una interfaz salen como caracteres sueltos o pares raros: "G", "*", "••",
    "|| O" —esos tres son los botones de navegación de Android—. Medido sobre
    999 líneas de un video real, 143 eran de esta clase y ninguna aportaba nada.

    El umbral es exigir dos caracteres alfanuméricos. Uno solo dejaba pasar los
    iconos que casualmente contienen una letra; dos los descarta y conserva
    igual las respuestas cortas legítimas de una interfaz: "Sí", "No", "OK",
    "93%".
    """
    return sum(c.isalnum() for c in line) >= 2


def read_screens(images: list[Path]) -> str:
    """Texto de una o varias pantallas, listo para entregarle al modelo."""
    blocks = []
    for i, image in enumerate(images, start=1):
        lines = read_text(image)
        if not lines:
            continue
        header = f"--- pantalla {i} ---\n" if len(images) > 1 else ""
        blocks.append(header + "\n".join(lines))
    return "\n\n".join(blocks)
