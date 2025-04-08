import numpy as np
from pydub import AudioSegment
import os

def calcular_spl(archivo_audio, volumen=1.0, presion_maxima=1.0, presion_referencia=20e-6):

    if not (0 < volumen <= 1):
        raise ValueError("El volumen debe estar entre 0 (exclusivo) y 1 (inclusive).")
    if presion_maxima <= 0 or presion_referencia <= 0:
        raise ValueError("presion_maxima y presion_referencia deben ser mayores que cero.")

    audio = AudioSegment.from_file(archivo_audio)
    muestras = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Calcular RMS normalizado (entre 0 y 1)
    rms = np.sqrt(np.mean(muestras ** 2)) / (2 ** (audio.sample_width * 8 - 1))

    # Ajustar por volumen y escalar a Pascales
    presion_rms = rms * volumen * presion_maxima

    # SPL en decibeles
    spl = 20 * np.log10(presion_rms / presion_referencia)
    return round(spl, 2)


# Carpeta que deseas explorar
directorio = "./"


# Iterar sobre todos los archivos en la carpeta
for archivo in os.listdir(directorio):
    ruta_completa = os.path.join(directorio, archivo)  # Obtener la ruta completa del archivo
    ""
    # Verificar si es un archivo (y no una subcarpeta)
    if os.path.isfile(ruta_completa ):
        if "wav" in ruta_completa:
            print(calcular_spl(archivo,0.0000894335192))
        # Aquí puedes realizar alguna operación, ""como leer su contenido, modificarlo, etc.
