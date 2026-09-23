"""
metrica_inteligibilidad_wer.py

Metrica propuesta: inteligibilidad mediante WER (Word Error Rate) vía ASR.

METODO
------
Se transcribe el audio con Whisper (modelo "small", forzado a espanol) y
se compara la transcripcion contra un texto de referencia mediante WER,
calculado con la libreria jiwer.

INTERPRETACION DEL VALOR
-------------------------
"wer" es una proporcion en [0, +inf) (puede superar 1.0 si hay muchas
inserciones respecto a una referencia corta).
MAS BAJO = MEJOR inteligibilidad (transcripcion mas fiel al texto real).
0.0 -> transcripcion identica al texto de referencia (caso ideal).
Valores mayores a ~0.3-0.5 sugieren errores de pronunciacion o contenido
significativos.

DEPENDENCIAS
------------
pip install openai-whisper "jiwer>=3.0" soundfile pandas

USO
---
    from metrica_inteligibilidad_wer import calcular_wer

    resultado = calcular_wer(
        "data/raw/daniel_sintetico_xtts.wav",
        "El Barcelona es el mejor equipo del mundo, sé que algún día "
        "serán campeones de Europa.",
    )
    print(resultado["wer"])            # ej. 0.0
    print(resultado["transcripcion"])  # lo que transcribio Whisper

    # Para reproducir el barrido completo del flujo de evaluacion
    # (solo sobre los audios sinteticos, unicos con texto de referencia
    # verificado; ver limitacion metodologica abajo):
    from metrica_inteligibilidad_wer import evaluar_dataset_completo
    evaluar_dataset_completo("data/raw")

LIMITACION METODOLOGICA IMPORTANTE
------------------------------------
WER requiere un texto de referencia verificado. Ese texto solo se conoce
con certeza para los 4 audios sinteticos generados con XTTS (esta
hardcodeado en scripts/generacion/generar_sintetico.py). Los audios
reales de data/raw/ NO tienen una transcripcion de referencia verificada
en el repositorio, por lo que esta funcion no debe usarse con una
referencia inventada para ellos: al no existir un texto verdadero,
cualquier "referencia" fabricada mediria la consistencia del propio ASR
consigo mismo, no un error de inteligibilidad real.

NOTA SOBRE LOS CASOS DE PRUEBA CONTROLADOS
--------------------------------------------
Los 3 casos de prueba pedidos en el objetivo especifico 2 del proyecto
("consigo misma", "mismo hablante", "hablantes distintos") estan
definidos para metricas que comparan AUDIO contra AUDIO (identidad,
prosodia). WER compara AUDIO contra TEXTO, una familia de metrica
distinta, por lo que ese esquema no aplica tal cual. Se definen en su
lugar 3 casos analogos, coherentes con la naturaleza de esta metrica:
  1. Determinismo: transcribir el mismo audio dos veces debe dar el
     mismo resultado (equivalente a "consigo mismo" para un modelo
     determinista).
  2. Texto de referencia correcto conocido (WER bajo esperado).
  3. Texto de referencia deliberadamente incorrecto (WER alto esperado),
     para confirmar que la metrica es sensible a la referencia usada.
"""

import os
import glob

import pandas as pd
import whisper
import jiwer

from audio_utils import cargar_audio

MODELO_WHISPER = "small"
IDIOMA = "es"

# Texto exacto usado para generar los audios *_sintetico_xtts.wav
# (tomado literalmente de scripts/generacion/generar_sintetico.py)
TEXTO_REFERENCIA_SINTETICOS = (
    "El Barcelona es el mejor equipo del mundo, "
    "sé que algún día serán campeones de Europa."
)

_modelo = None

_transformacion = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        _modelo = whisper.load_model(MODELO_WHISPER)
    return _modelo


def calcular_wer(ruta_audio, texto_referencia):
    """
    Transcribe un audio y calcula su WER contra un texto de referencia.

    Parametros
    ----------
    ruta_audio : str
        Ruta al archivo .wav. Debe contar con un texto de referencia
        VERIFICADO (ver limitacion metodologica arriba); no usar con
        audios cuyo contenido real se desconoce.
    texto_referencia : str
        Texto exacto que se espera que diga el audio.

    Devuelve
    --------
    dict con:
        "wer" : float en [0, inf). Mas bajo = mejor inteligibilidad.
        "transcripcion" : str, lo transcrito por Whisper
    """
    modelo = _cargar_modelo()
    audio = cargar_audio(ruta_audio, como_tensor=False)  # array 1D, 16kHz
    resultado = modelo.transcribe(audio, language=IDIOMA, fp16=False)
    transcripcion = resultado["text"].strip()

    wer = jiwer.wer(
        texto_referencia,
        transcripcion,
        reference_transform=_transformacion,
        hypothesis_transform=_transformacion,
    )

    return {"wer": float(wer), "transcripcion": transcripcion}


def evaluar_dataset_completo(carpeta_audios="data/raw"):
    """
    Reproduce el flujo de evaluacion agregado del repositorio (equivalente
    al de calcular_wer.py original): transcribe todos los audios de
    carpeta_audios, calcula WER solo para los sinteticos (unicos con
    texto de referencia verificado) y transcribe los reales solo con
    fines de inspeccion cualitativa (sin WER asociado).

    Parametros
    ----------
    carpeta_audios : str
        Carpeta con archivos .wav nombrados como "<hablante>_<...>.wav",
        donde los sinteticos contienen "sintetico" o "xtts" en el nombre.

    Devuelve
    --------
    pandas.DataFrame con archivo, hablante, tipo, transcripcion y WER
    (WER = None para los audios reales).
    """
    rutas_audios = sorted(glob.glob(os.path.join(carpeta_audios, "*.wav")))
    registros = []

    print("1. Cargando modelo Whisper...")
    _cargar_modelo()

    print("2. Transcribiendo audios de la carpeta...")
    for ruta in rutas_audios:
        nombre = os.path.basename(ruta).lower()
        partes = nombre.replace(".wav", "").split("_")
        hablante = partes[0]
        tipo = "sintetico" if ("sintetico" in nombre or "xtts" in nombre) else "real"

        if tipo == "sintetico":
            r = calcular_wer(ruta, TEXTO_REFERENCIA_SINTETICOS)
            registros.append({"archivo": nombre, "hablante": hablante, "tipo": tipo,
                               "transcripcion": r["transcripcion"], "WER": r["wer"]})
        else:
            audio = cargar_audio(ruta, como_tensor=False)
            transcripcion = _cargar_modelo().transcribe(audio, language=IDIOMA, fp16=False)["text"].strip()
            registros.append({"archivo": nombre, "hablante": hablante, "tipo": tipo,
                               "transcripcion": transcripcion, "WER": None})

    df = pd.DataFrame(registros)

    print("\n3. Calculando WER para audios sinteticos...")
    df_sint = df[df["tipo"] == "sintetico"]
    print("Audios SINTETICOS (WER calculado):")
    print(df_sint[["archivo", "hablante", "transcripcion", "WER"]].to_string(index=False))
    if len(df_sint) > 0:
        print(f"\nWER promedio (sinteticos): {df_sint['WER'].mean() * 100:.2f}%")

    df_real = df[df["tipo"] == "real"]
    print("\nAudios REALES (sin WER, solo transcripcion):")
    print(df_real[["archivo", "hablante", "transcripcion"]].to_string(index=False))

    return df


if __name__ == "__main__":
    AUDIO_DANIEL = "data/raw/daniel_sintetico_xtts.wav"
    AUDIO_FELIPE = "data/raw/felipe_sintetico_xtts.wav"

    print("--- Pruebas controladas: metrica_inteligibilidad_wer ---\n")

    print("Caso 1: determinismo (transcribir el mismo audio dos veces)")
    r1 = calcular_wer(AUDIO_DANIEL, TEXTO_REFERENCIA_SINTETICOS)
    r2 = calcular_wer(AUDIO_DANIEL, TEXTO_REFERENCIA_SINTETICOS)
    print(f"  -> transcripcion 1: {r1['transcripcion']}")
    print(f"  -> transcripcion 2: {r2['transcripcion']}")
    print(f"  -> iguales: {r1['transcripcion'] == r2['transcripcion']}\n")

    print("Caso 2: texto de referencia CORRECTO, en DOS audios distintos (esperado: WER bajo en ambos)")
    r_daniel = calcular_wer(AUDIO_DANIEL, TEXTO_REFERENCIA_SINTETICOS)
    print(f"  -> daniel: {r_daniel}")
    r_felipe = calcular_wer(AUDIO_FELIPE, TEXTO_REFERENCIA_SINTETICOS)
    print(f"  -> felipe: {r_felipe}\n")

    print("Caso 3: texto de referencia deliberadamente INCORRECTO (esperado: WER alto)")
    r = calcular_wer(AUDIO_DANIEL, "El clima de hoy está muy nublado en Heredia.")
    print(f"  -> {r}\n")

    print("--- Barrido completo (flujo de evaluacion del repositorio) ---\n")
    evaluar_dataset_completo("data/raw")
