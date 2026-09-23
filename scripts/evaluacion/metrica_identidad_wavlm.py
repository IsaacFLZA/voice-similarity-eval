"""
metrica_identidad_wavlm.py

Metrica propuesta: identidad del hablante mediante embeddings WavLM.

METODO
------
Se extrae un embedding de 512 dimensiones por audio con el modelo
microsoft/wavlm-base-plus-sv (WavLM afinado para verificacion de
hablante), y se compara un par de audios mediante similitud coseno
entre sus embeddings.

INTERPRETACION DEL VALOR
-------------------------
"similitud_coseno" esta en el rango [-1, 1].
MAS ALTO = MAYOR similitud (mas probable que sea el mismo hablante).
Valores cercanos a 1.0 -> mismo hablante muy probable.
Valores cercanos a 0.0 o negativos -> hablantes distintos.

DEPENDENCIAS
------------
pip install transformers==4.36.2 torch==2.1.0 torchaudio==2.1.0 soundfile scikit-learn matplotlib

(la version de transformers esta acotada deliberadamente: versiones mas
nuevas requieren torch>=2.5, incompatible con la version de torch fijada
para el resto del repositorio; ver bitacora Semana 4-5 para el detalle
de esta incompatibilidad)

USO
---
    from metrica_identidad_wavlm import calcular_similitud_wavlm

    resultado = calcular_similitud_wavlm("data/raw/daniel_ref1.wav",
                                          "data/raw/daniel_ref2.wav")
    print(resultado["similitud_coseno"])  # ej. 0.87

    # Para reproducir el barrido completo del flujo de evaluacion,
    # equivalente al de calcular_eer_general.py pero con WavLM:
    from metrica_identidad_wavlm import evaluar_dataset_completo
    evaluar_dataset_completo("data/raw")

LIMITACION CONOCIDA
--------------------
Con transformers==4.36.2, los pesos de la capa pos_conv_embed del modelo
no se cargan desde el checkpoint (desajuste de nomenclatura de parametros
entre versiones) y quedan inicializados aleatoriamente. El resto de la
arquitectura carga correctamente. Ver bitacora Semana 4, sesion 05/09.
"""

import os
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

from audio_utils import cargar_audio

MODEL_NAME = "microsoft/wavlm-base-plus-sv"

_feature_extractor = None
_model = None


def _cargar_modelo():
    """Carga el modelo una sola vez (patron singleton a nivel de modulo)."""
    global _feature_extractor, _model
    if _model is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _model = WavLMForXVector.from_pretrained(MODEL_NAME)
        _model.eval()
    return _feature_extractor, _model


def _extraer_embedding(ruta_audio):
    feature_extractor, model = _cargar_modelo()
    audio = cargar_audio(ruta_audio, como_tensor=False)  # array 1D, 16kHz
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        salida = model(**inputs)
    return salida.embeddings.squeeze().cpu().numpy()


def calcular_similitud_wavlm(ruta_audio_1, ruta_audio_2):
    """
    Calcula la similitud de identidad de hablante entre dos audios.

    Parametros
    ----------
    ruta_audio_1, ruta_audio_2 : str
        Rutas a los archivos .wav a comparar.

    Devuelve
    --------
    dict con:
        "similitud_coseno" : float en [-1, 1]. Mas alto = mas similar.
        "modelo" : str, nombre del modelo usado
    """
    emb1 = _extraer_embedding(ruta_audio_1)
    emb2 = _extraer_embedding(ruta_audio_2)
    similitud = float(cosine_similarity([emb1], [emb2])[0][0])
    return {"similitud_coseno": similitud, "modelo": MODEL_NAME}


def evaluar_dataset_completo(carpeta_audios="data/raw", guardar_grafica=True):
    """
    Reproduce el flujo de evaluacion agregado del repositorio (equivalente
    al de calcular_eer_general.py, pero usando la metrica WavLM):
    compara todos los pares de audios de carpeta_audios, clasifica cada
    par como genuino / impostor natural / ataque sintetico, y calcula el
    Equal Error Rate (EER) global.

    Parametros
    ----------
    carpeta_audios : str
        Carpeta con archivos .wav nombrados como "<hablante>_<...>.wav",
        donde los sinteticos contienen "sintetico" o "xtts" en el nombre.
    guardar_grafica : bool
        Si True, guarda la curva ROC como 'curva_roc_wavlm.png'.

    Devuelve
    --------
    dict con "eer" (float, 0-1) y "umbral_optimo" (float).
    """
    rutas_audios = sorted(glob.glob(os.path.join(carpeta_audios, "*.wav")))
    metadata = []
    embeddings = []

    print("1. Cargando el modelo WavLM...")
    _cargar_modelo()

    print("2. Procesando audios de la carpeta y extrayendo embeddings...")
    for ruta in rutas_audios:
        nombre = os.path.basename(ruta).lower()
        partes = nombre.replace(".wav", "").split("_")
        identidad = partes[0]
        tipo = "sintetico" if ("sintetico" in nombre or "xtts" in nombre) else "real"
        metadata.append({"id": identidad, "tipo": tipo, "archivo": nombre})
        embeddings.append(_extraer_embedding(ruta))

    print("3. Calculando cruces y clasificando accesos...")
    y_true, y_scores = [], []
    for i in range(len(metadata)):
        for j in range(i + 1, len(metadata)):
            fi, fj = metadata[i], metadata[j]
            similitud = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

            if fi["id"] == fj["id"] and fi["tipo"] == "real" and fj["tipo"] == "real":
                y_true.append(1); y_scores.append(similitud)
            elif fi["id"] != fj["id"] and fi["tipo"] == "real" and fj["tipo"] == "real":
                y_true.append(0); y_scores.append(similitud)
            elif fi["id"] == fj["id"] and (fi["tipo"] == "sintetico" or fj["tipo"] == "sintetico"):
                y_true.append(0); y_scores.append(similitud)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer_threshold = float(thresholds[eer_idx])
    eer = float(fpr[eer_idx])

    print("\n4. Calculando Umbral Estadistico (EER)...")
    print(f"-> Se evaluaron {len(y_scores)} cruces validos.")
    print(f"Umbral Optimo (Threshold): {eer_threshold:.4f}")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")

    if guardar_grafica:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="seagreen", lw=2, label=f"Curva ROC WavLM (EER = {eer*100:.2f}%)")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("Tasa de Falsa Aceptacion (FPR)")
        plt.ylabel("Tasa de Verdadera Aceptacion (TPR)")
        plt.title("Curva ROC - Identidad del hablante con WavLM (propuesta)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig("curva_roc_wavlm.png", dpi=300)
        print("Grafica guardada como 'curva_roc_wavlm.png'.")

    return {"eer": eer, "umbral_optimo": eer_threshold}


if __name__ == "__main__":
    # Pruebas basicas con casos controlados, tal como exige el objetivo
    # especifico 2 del proyecto.
    print("--- Pruebas controladas: metrica_identidad_wavlm ---\n")

    print("Caso 1: audio consigo mismo (esperado: similitud ~1.0)")
    r = calcular_similitud_wavlm("data/raw/daniel_ref1.wav", "data/raw/daniel_ref1.wav")
    print(f"  -> {r}\n")

    print("Caso 2: mismo hablante, audios distintos (esperado: similitud alta)")
    r = calcular_similitud_wavlm("data/raw/daniel_ref1.wav", "data/raw/daniel_ref2.wav")
    print(f"  -> {r}\n")

    print("Caso 3: hablantes distintos (esperado: similitud mas baja que en el caso 2)")
    r = calcular_similitud_wavlm("data/raw/daniel_ref1.wav", "data/raw/felipe_test1.wav")
    print(f"  -> {r}\n")

    print("--- Barrido completo (flujo de evaluacion del repositorio) ---\n")
    evaluar_dataset_completo("data/raw")
