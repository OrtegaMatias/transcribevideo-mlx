"""Prompts del motor.

Dos decisiones deliberadas, ambas para contrarrestar sesgos conocidos del
modelo:

1. `texto_en_pantalla` se pide antes que cualquier campo interpretativo, y con
   una prohibición explícita de completar con lo oído. Al darle el audio junto
   con la imagen, el modelo tiende a "leer" en pantalla lo que solo escuchó.

2. La pregunta de continuidad está formulada al revés de lo intuitivo. A un
   modelo al que se le pregunta "¿te falta contexto?" casi siempre responde que
   sí; se le pregunta si la unidad **se entiende sola** y se le exige un corte
   literal para responder que no.
"""

CHUNK_SYSTEM = """\
Analizas una grabación de pantalla. Recibes una o más capturas consecutivas de \
la pantalla y la transcripción de lo que el narrador dijo mientras esas \
pantallas estaban visibles.

Devuelves EXCLUSIVAMENTE un objeto JSON válido, sin texto antes ni después, sin \
bloques de código, con exactamente estas claves:

{
  "titulo": string,
  "texto_en_pantalla": string,
  "elementos_ui": [string],
  "narracion": string,
  "sintesis": string,
  "se_entiende_sola": boolean,
  "motivo": string
}

Reglas:

- "titulo": nombre corto de la pantalla, tal como aparece en ella si es posible.
- "texto_en_pantalla": SOLO texto que puedas LEER literalmente en las imágenes. \
Transcríbelo respetando etiquetas y valores. Si algo se menciona en el audio \
pero NO está escrito en la imagen, NO lo incluyas. No completes, no infieras, \
no corrijas. Si la pantalla no tiene texto legible, deja el string vacío.
- "elementos_ui": controles visibles (botones, campos, pestañas, tablas, menús).
- "narracion": qué dijo el narrador en este tramo, resumido y en sus términos.
- "sintesis": qué se está haciendo o explicando aquí, cruzando lo que se ve con \
lo que se dice. Es el único campo donde puedes relacionar ambas fuentes.
- "se_entiende_sola": true si este tramo forma una unidad comprensible por sí \
misma. Responde false SOLO si una oración o un procedimiento queda literalmente \
truncado y se completa en el tramo siguiente. Ante la duda, true.
- "motivo": una frase justificando el valor de "se_entiende_sola".

Responde en el mismo idioma del audio."""


def chunk_user_prompt(window_label: str, window_audio: str,
                      previous_tail: str, next_head: str,
                      n_screens: int) -> str:
    """Prompt del analizador por tramo.

    El contexto vecino va marcado explícitamente como no-analizable: sirve para
    juzgar la continuidad, no para describirse. Sin él, el modelo no tiene forma
    de saber si la frase que quedó colgando se completa después o no existe.
    """
    plural = "las pantallas" if n_screens > 1 else "la pantalla"
    parts = []

    if previous_tail:
        parts.append(
            "CONTEXTO PREVIO (no lo analices; úsalo solo para juzgar continuidad):\n"
            f"{previous_tail}")

    parts.append(
        f"=== AUDIO DE ESTE TRAMO ({window_label}) ===\n"
        f"{window_audio or '(sin habla en este tramo)'}\n"
        "=== FIN DEL TRAMO ===")

    if next_head:
        parts.append(
            "CONTEXTO SIGUIENTE (no lo analices; úsalo solo para juzgar continuidad):\n"
            f"{next_head}")

    parts.append(
        f"Analiza {plural} adjunta{'s' if n_screens > 1 else ''} junto con el "
        "audio del tramo y devuelve el JSON.")
    return "\n\n".join(parts)


REPORT_SYSTEM = """\
Redactas el informe final de una grabación de pantalla que ya fue analizada \
tramo por tramo. Recibes, en orden cronológico, el análisis de cada pantalla \
(qué se veía y qué se decía) y la transcripción completa del audio.

Escribe un informe en Markdown, en el idioma del contenido, con esta estructura:

## Resumen
Tres a seis frases: de qué trata la grabación, qué sistema se muestra y qué se \
logra en ella.

## Qué se hace, paso a paso
Los pasos o temas en orden, cada uno con su marca de tiempo. Agrupa tramos \
contiguos que sean parte del mismo procedimiento en vez de repetirlos.

## Datos que aparecen en pantalla
Los valores concretos leídos de la interfaz (identificadores, cifras, estados, \
nombres de campos), atribuidos a la pantalla donde aparecen. Este apartado se \
alimenta SOLO de lo que fue leído en pantalla, nunca de lo dicho en el audio.

## Observaciones
Contradicciones entre lo que se dice y lo que se muestra, pasos que quedan a \
medias, o supuestos que el narrador da por sabidos. Si no hay nada que señalar, \
escribe "Sin observaciones."

Reglas: no inventes datos que no estén en el material. Si algo es ambiguo, dilo. \
No repitas literalmente la transcripción: sintetiza."""


def report_user_prompt(chunks_block: str, transcript: str) -> str:
    return (
        "=== ANÁLISIS POR PANTALLA ===\n"
        f"{chunks_block}\n\n"
        "=== TRANSCRIPCIÓN COMPLETA DEL AUDIO ===\n"
        f"{transcript}\n\n"
        "Redacta el informe.")
