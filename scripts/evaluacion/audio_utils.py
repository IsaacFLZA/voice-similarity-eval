"""
audio_utils.py

Utilidad compartida por los modulos de metricas nuevas (WavLM, U3D, WER).
No es parte del repositorio original ni lo modifica: es un archivo nuevo,
creado para evitar duplicar la misma logica de carga de audio en cada
modulo.

Carga los audios con soundfile (en vez de torchaudio.load()) para no
depender de torchcodec/FFmpeg, cuya instalacion en Windows requirio
resolver varios problemas de entorno documentados en la bitacora de la
Semana 4.
"""

import soundfile as sf
import numpy as np
import torch
import torchaudio.functional as AF

FS_OBJETIVO = 16000


def cargar_audio(ruta, fs_objetivo=FS_OBJETIVO, como_tensor=True):
    """
    Carga un audio y lo resamplea a fs_objetivo (16 kHz por defecto).

    Parametros
    ----------
    ruta : str
        Ruta al archivo .wav
    fs_objetivo : int
        Frecuencia de muestreo deseada
    como_tensor : bool
        Si True, devuelve un tensor de PyTorch de forma (1, 1, n_muestras),
        listo para modelos tipo WavLM/HuBERT. Si False, devuelve un
        array de numpy 1D (mono), formato esperado por Whisper.

    Devuelve
    --------
    torch.Tensor o np.ndarray, segun como_tensor
    """
    señal, fs = sf.read(ruta, dtype="float32")

    if señal.ndim > 1:
        señal = señal.mean(axis=1)  # a mono si viniera estereo

    if como_tensor:
        tensor = torch.from_numpy(señal).unsqueeze(0)  # (1, n_muestras)
        if fs != fs_objetivo:
            tensor = AF.resample(tensor, fs, fs_objetivo)
        return tensor.unsqueeze(0)  # (1, 1, n_muestras)
    else:
        if fs != fs_objetivo:
            import librosa
            señal = librosa.resample(señal, orig_sr=fs, target_sr=fs_objetivo)
        return señal
