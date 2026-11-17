"""
metrica_prosodia_u3d.py

Metrica propuesta: prosodia y ritmo mediante U3D (Unit Duration
Distribution Distance), tal como la definen Carbonneau et al.
(arXiv:2507.02176) y su implementacion oficial:
https://github.com/ubisoft/ubisoft-laforge-spkrid

METODO
------
Cada audio se segmenta en unidades tipo-fonema (Approximant, Fricative,
Nasal, Silence, Stop, Vowel) mediante HuBERT Soft + el segmentador
URhythmic (8 clusters, no supervisado). Se obtiene asi, por audio, una
lista de duraciones (ms) por categoria. Dos audios se comparan calculando
la distancia de Wasserstein entre sus distribuciones de duracion,
categoria por categoria.

INTERPRETACION DEL VALOR
-------------------------
Las distancias estan en milisegundos (ms), sin techo definido (dependen
de la duracion de los audios comparados).
MAS BAJO = ritmos MAS PARECIDOS entre los dos audios.
0 ms -> distribuciones de duracion identicas (caso ideal: mismo audio).
Valores altos -> los dos audios difieren notablemente en cuanto duran
sus fonemas de esa categoria.

DEPENDENCIAS
------------
pip install torch==2.1.0 torchaudio==2.1.0 scipy pandas soundfile numpy
(HuBERT Soft y el segmentador URhythmic se descargan automaticamente via
torch.hub la primera vez que se ejecuta; no se instalan con pip)

USO
---
    from metrica_prosodia_u3d import calcular_distancia_u3d

    resultado = calcular_distancia_u3d("data/raw/daniel_ref1.wav",
                                        "data/raw/daniel_sintetico_xtts.wav")
    print(resultado["distancias_por_grupo"])
    print(resultado["promedio"])

    # Para reproducir el barrido completo del flujo de evaluacion,
    # comparando mismo_hablante vs hablantes_distintos como linea base:
    from metrica_prosodia_u3d import evaluar_dataset_completo
    evaluar_dataset_completo("data/raw")

LIMITACION CONOCIDA
--------------------
El criterio de exclusion de silencios (heredado de la implementacion
oficial) solo descarta el primer y ultimo segmento de silencio del
audio, no los silencios internos. En audios sin recorte previo de VAD
(como los de data/raw/), esto puede inflar fuertemente la categoria
"silence" y no debe interpretarse como diferencia real de ritmo del
habla. Ver bitacora Semana 5, sesion 12/09.
"""

import os
import glob
import itertools

import torch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from itertools import pairwise

from audio_utils import cargar_audio

NUM_CLUSTERS = 8
HOP_LENGTH_S = 0.02

_hubert = None
_segmenter = None


def _cargar_modelos():
    global _hubert, _segmenter
    if _hubert is None:
        _hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
        _hubert.eval()
        _segmenter = torch.hub.load(
            "bshall/urhythmic:main", "segmenter", num_clusters=NUM_CLUSTERS, trust_repo=True
        )
    return _hubert, _segmenter


@torch.inference_mode()
def _extraer_duraciones_por_grupo(ruta_audio):
    """Devuelve un dict {grupo: [duraciones_ms]} para un audio."""
    hubert, segmenter = _cargar_modelos()
    audio_tensor = cargar_audio(ruta_audio, como_tensor=True)

    units = hubert.units(audio_tensor)
    logits = hubert.logits(units)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze().cpu().numpy()

    segmentos, limites = segmenter(log_probs)

    duraciones = {}
    n = len(segmentos)
    for i, (segmento, (a, b)) in enumerate(zip(segmentos, pairwise(limites))):
        nombre = segmento.name.lower()
        if segmento.name == "SILENCE" and (i == 0 or i == n - 1):
            continue  # silencio de borde, se descarta (ver limitacion documentada)
        duraciones.setdefault(nombre, []).append((b - a) * HOP_LENGTH_S * 1000)

    return duraciones


def calcular_distancia_u3d(ruta_audio_1, ruta_audio_2):
    """
    Calcula la distancia U3D entre dos audios: distancia de Wasserstein
    entre sus distribuciones de duracion, por categoria tipo-fonema.

    Parametros
    ----------
    ruta_audio_1, ruta_audio_2 : str

    Devuelve
    --------
    dict con:
        "distancias_por_grupo" : dict {grupo: distancia_ms}. Mas bajo = mas parecido.
        "promedio" : float, promedio de las distancias por grupo
    """
    dur1 = _extraer_duraciones_por_grupo(ruta_audio_1)
    dur2 = _extraer_duraciones_por_grupo(ruta_audio_2)

    grupos_comunes = sorted(set(dur1) & set(dur2))
    distancias = {
        g: float(wasserstein_distance(dur1[g], dur2[g])) for g in grupos_comunes
    }
    promedio = float(np.mean(list(distancias.values()))) if distancias else float("nan")

    return {"distancias_por_grupo": distancias, "promedio": promedio}


def evaluar_dataset_completo(carpeta_audios="data/raw"):
    """
    Reproduce el flujo de evaluacion agregado del repositorio (equivalente
    al de calcular_u3d.py original): para cada categoria tipo-fonema,
    calcula la distancia promedio en los casos "mismo_hablante" (real vs.
    sintetico del mismo hablante) y "hablantes_distintos" (real vs. real
    entre hablantes diferentes, como linea base).

    Parametros
    ----------
    carpeta_audios : str
        Carpeta con archivos .wav nombrados como "<hablante>_<...>.wav".

    Devuelve
    --------
    pandas.DataFrame con los resultados por grupo tipo-fonema.
    """
    rutas_audios = sorted(glob.glob(os.path.join(carpeta_audios, "*.wav")))
    registros = []

    print("1. Cargando modelos HuBERT Soft y segmentador URhythmic...")
    _cargar_modelos()

    print("2. Procesando audios y extrayendo duraciones por categoria...")
    for ruta in rutas_audios:
        nombre = os.path.basename(ruta).lower()
        partes = nombre.replace(".wav", "").split("_")
        hablante = partes[0]
        tipo = "sintetico" if ("sintetico" in nombre or "xtts" in nombre) else "real"

        for grupo, duraciones in _extraer_duraciones_por_grupo(ruta).items():
            for d in duraciones:
                registros.append((hablante, tipo, grupo, d))

    df = pd.DataFrame(registros, columns=["hablante", "tipo", "grupo", "duracion_ms"])
    print(f"-> Se extrajeron {len(df)} segmentos en total.")

    print("\n3. Calculando distancias de Wasserstein por categoria...")
    hablantes = sorted(df["hablante"].unique())
    grupos = sorted(df["grupo"].unique())
    resultados = []

    for grupo in grupos:
        df_grupo = df[df["grupo"] == grupo]

        distancias_mismo = []
        for hablante in hablantes:
            dur_real = df_grupo[(df_grupo["hablante"] == hablante) & (df_grupo["tipo"] == "real")]["duracion_ms"]
            dur_sint = df_grupo[(df_grupo["hablante"] == hablante) & (df_grupo["tipo"] == "sintetico")]["duracion_ms"]
            if len(dur_real) > 0 and len(dur_sint) > 0:
                distancias_mismo.append(wasserstein_distance(dur_real, dur_sint))

        distancias_distintos = []
        for h1, h2 in itertools.combinations(hablantes, 2):
            dur_h1 = df_grupo[(df_grupo["hablante"] == h1) & (df_grupo["tipo"] == "real")]["duracion_ms"]
            dur_h2 = df_grupo[(df_grupo["hablante"] == h2) & (df_grupo["tipo"] == "real")]["duracion_ms"]
            if len(dur_h1) > 0 and len(dur_h2) > 0:
                distancias_distintos.append(wasserstein_distance(dur_h1, dur_h2))

        resultados.append({
            "grupo": grupo,
            "mismo_hablante (real vs sintetico)": np.mean(distancias_mismo) if distancias_mismo else np.nan,
            "hablantes_distintos (real vs real, linea base)": np.mean(distancias_distintos) if distancias_distintos else np.nan,
        })

    tabla = pd.DataFrame(resultados).round(2)
    print("\n4. Resultados finales por categoria tipo-fonema:")
    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    print("--- Pruebas controladas: metrica_prosodia_u3d ---\n")

    print("Caso 1: audio consigo mismo (esperado: distancias ~0 en todos los grupos)")
    r = calcular_distancia_u3d("data/raw/daniel_ref1.wav", "data/raw/daniel_ref1.wav")
    print(f"  -> {r}\n")

    print("Caso 2: mismo hablante, real vs sintetico (esperado: distancias > 0)")
    r = calcular_distancia_u3d("data/raw/daniel_ref1.wav", "data/raw/daniel_sintetico_xtts.wav")
    print(f"  -> {r}\n")

    print("Caso 3: hablantes distintos, ambos reales (linea base de variacion natural)")
    r = calcular_distancia_u3d("data/raw/daniel_ref1.wav", "data/raw/felipe_test1.wav")
    print(f"  -> {r}\n")

    print("--- Barrido completo (flujo de evaluacion del repositorio) ---\n")
    evaluar_dataset_completo("data/raw")
