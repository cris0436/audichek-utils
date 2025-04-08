import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

def generate_stereo_wav_file(frequency, volume, duration, filename):
    sample_rate = 44100  # Tasa de muestreo
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  
    # Generar señal sinusoidal para ambos canales
    audio_data = volume * np.sin(2 * np.pi * frequency * t)
    # Crear un array estéreo
    audio_stereo = np.column_stack((audio_data, audio_data)).astype(np.float32)  # Cambiamos a float32
    # Guardar el archivo como WAV con alta calidad
    write(filename, sample_rate, (audio_stereo * 32767).astype(np.int16))  # Convertimos solo al guardar
# Ejemplo de uso
frequencies = [8000, 10000, 12000, 15000, 16000, 17000, 18000, 19000, 20000]

for i in frequencies:
    generate_stereo_wav_file(i, 1, 15, f'{i}.wav')



