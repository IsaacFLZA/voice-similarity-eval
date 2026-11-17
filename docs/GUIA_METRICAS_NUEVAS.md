# Guía de uso de las métricas nuevas

Esta guía describe los módulos agregados a `scripts/evaluacion/` para ampliar la
evaluación de voces naturales y sintéticas: identidad con WavLM, ritmo con U3D e
inteligibilidad con WER. Los scripts heredados (`calcular_eer_general.py`,
`calcular_mcd.py`, `calcular_mos_nisqa.py`) no se modificaron.

## Archivos

| Archivo | Dimensión | Función principal |
|---|---|---|
| `audio_utils.py` | (compartido) | `cargar_audio(ruta, como_tensor=True)` |
| `metrica_identidad_wavlm.py` | Identidad del hablante | `calcular_similitud_wavlm(audio_1, audio_2)` |
| `metrica_prosodia_u3d.py` | Ritmo y prosodia | `calcular_distancia_u3d(audio_1, audio_2)` |
| `metrica_inteligibilidad_wer.py` | Inteligibilidad | `calcular_wer(audio, texto_referencia)` |

`audio_utils.py` lee los `.wav` con `soundfile`, convierte a mono y remuestrea a
16 kHz, sin depender de `torchaudio.load()` ni de FFmpeg.

## Instalación

Las versiones están en `requirements_metricas_nuevas.txt`, que corresponde al entorno
donde se desarrollaron los módulos (Python 3.11). El `requirements.txt` general del
repositorio no incluye las bibliotecas de las métricas nuevas.

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements_metricas_nuevas.txt
```

Notas:

- Se recomienda Python 3.11. `torch==2.1.0` no tiene versiones para Python 3.12 o superior.
- En Linux, `pip` descarga por defecto una versión de PyTorch con librerías de GPU
  (varios GB). Para evitarlo, instalar antes `torch` y `torchaudio` desde el índice CPU:
  `pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu`.
- `transformers` está acotado a 4.36.2 porque las versiones más recientes exigen
  una versión de `torch` incompatible con la fijada para el resto del repositorio.
- La primera ejecución descarga los modelos: WavLM (~405 MB, Hugging Face),
  HuBERT Soft y URhythmic (~361 MB, `torch.hub`) y Whisper *small* (~461 MB).

## Convenciones de los datos

Los módulos que recorren una carpeta (`evaluar_dataset_completo`) esperan archivos
`.wav` nombrados `<hablante>_<descripcion>.wav`:

- El hablante es el texto antes del primer `_` (por ejemplo, `daniel`).
- Los audios sintéticos deben contener `sintetico` o `xtts` en el nombre.
- Los demás se consideran grabaciones reales.

Para identidad se comparan todos los pares y se clasifican como **genuino** (mismo
hablante, ambos reales), **impostor** (hablantes distintos, ambos reales) o **ataque**
(mismo hablante, uno sintético). Los impostores y ataques son los negativos.

## Uso por métrica

Ejecutar siempre desde la raíz del repositorio. Para importar desde otro script:

```python
import sys
sys.path.append("scripts/evaluacion")
```

### Identidad con WavLM

```python
from metrica_identidad_wavlm import calcular_similitud_wavlm, evaluar_dataset_completo

r = calcular_similitud_wavlm("data/raw/daniel_ref1.wav", "data/raw/daniel_ref2.wav")
print(r["similitud_coseno"], r["modelo"])

evaluar_dataset_completo("data/raw")   # EER, umbral y curva ROC (curva_roc_wavlm.png)
```

- **Interpretación:** similitud coseno en [-1, 1]; mayor = más probable que sea el mismo hablante.
- **Importante:** la escala de WavLM no es comparable con la de ECAPA-TDNN
  (sus similitudes son más altas). Comparar modelos mediante el EER, no con los umbrales.

### Ritmo con U3D

```python
from metrica_prosodia_u3d import calcular_distancia_u3d, evaluar_dataset_completo

r = calcular_distancia_u3d("data/raw/daniel_ref1.wav", "data/raw/daniel_sintetico_xtts.wav")
print(r["distancias_por_grupo"])   # ms, por categoría fonética
print(r["promedio"])

evaluar_dataset_completo("data/raw")   # tabla "mismo hablante" frente a "línea base"
```

- **Interpretación:** distancia de Wasserstein en milisegundos; menor = ritmos más parecidos; 0 = idénticos.
- **Importante:** la categoría `silence` puede dominar el promedio si los audios no tienen
  recorte previo de silencios. Reportar los valores por categoría o el promedio sin `silence`.

### Inteligibilidad con WER

```python
from metrica_inteligibilidad_wer import calcular_wer, evaluar_dataset_completo

r = calcular_wer(
    "data/raw/daniel_sintetico_xtts.wav",
    "El Barcelona es el mejor equipo del mundo, sé que algún día serán campeones de Europa.",
)
print(r["wer"], r["transcripcion"])

evaluar_dataset_completo("data/raw")   # WER de los sintéticos y transcripción de los reales
```

- **Interpretación:** proporción de error de palabras; menor = mejor; 0 = transcripción idéntica a la referencia.
  Puede superar 1.0 si hay muchas inserciones.
- **Importante:** el WER solo tiene sentido con un texto de referencia verificado. Antes de
  calcularlo, comprobar que el audio realmente diga ese texto: en el conjunto de ejemplo, el clon
  de Paula pronuncia una oración distinta y su WER (43.75 %) no refleja baja inteligibilidad.
  `evaluar_dataset_completo` solo calcula WER para los sintéticos, usando el texto con el que se generaron.

## Pruebas con casos controlados

Cada módulo, ejecutado directamente, corre sus pruebas y el barrido completo:

```bash
python scripts/evaluacion/metrica_identidad_wavlm.py
python scripts/evaluacion/metrica_prosodia_u3d.py
python scripts/evaluacion/metrica_inteligibilidad_wer.py
```

Resultados esperados con los audios de `data/raw/`:

- Un audio comparado consigo mismo: similitud 1.0 (WavLM) y distancias 0 (U3D).
- Mismo hablante: similitud alta en WavLM (≈ 0.98 entre `daniel_ref1` y `daniel_ref2`).
- Hablantes distintos: similitud menor (≈ 0.74 entre `daniel_ref1` y `felipe_test1`).
- WER: dos transcripciones idénticas del mismo audio; WER bajo con la referencia correcta y alto con una incorrecta.

## Limitaciones conocidas

- **WavLM:** con `transformers==4.36.2` se observó una advertencia de que los pesos de la capa
  `pos_conv_embed` no se cargan desde el *checkpoint* y quedan inicializados al azar.
- **U3D:** el criterio heredado solo descarta el primer y el último segmento de silencio.
- **WER:** solo se puede calcular con texto de referencia verificado.
- **MCD (script heredado):** sus valores cambian entre entornos; usarlo como distancia relativa.

## Cómo agregar una nueva métrica

Para mantener una interfaz consistente, un módulo nuevo en `scripts/evaluacion/` debe:

1. Llamarse `metrica_<dimension>_<nombre>.py` y usar `audio_utils.cargar_audio` para leer audio.
2. Exponer una función `calcular_<nombre>(...)` que reciba rutas de audio (y un texto, si aplica)
   y devuelva un diccionario con el valor principal y, si corresponde, metadatos.
3. Cargar el modelo una sola vez (por ejemplo, con una variable de módulo inicializada al primer uso).
4. Incluir `evaluar_dataset_completo(carpeta)`, que recorra los audios con las convenciones de arriba
   e imprima los resultados en pasos numerados, como los scripts existentes.
5. Documentar en el docstring: método, **interpretación del valor** (rango y si mayor o menor es mejor),
   dependencias, ejemplo de uso y limitaciones.
6. Incluir en el bloque `if __name__ == "__main__":` pruebas con casos controlados (consigo mismo,
   mismo hablante, hablantes distintos, o el equivalente si la métrica no compara dos audios).
7. Agregar sus dependencias a `requirements_metricas_nuevas.txt`.
