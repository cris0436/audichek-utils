import joblib
import pandas as pd

# Cargar el modelo y el scaler desde los archivos
modelo_cargado = joblib.load('modelo_lineal_entrenado.pkl')
scaler_cargado = joblib.load('scaler_entrenado.pkl')

# Las mismas frecuencias que usaste durante el entrenamiento
frecuencias = [
    'frecuencia_250', 'frecuencia_1000', 'frecuencia_4000',
    'frecuencia_9000', 'frecuencia_10000', 'frecuencia_15000',
    'frecuencia_16000', 'frecuencia_17000', 'frecuencia_18000',
    'frecuencia_19000', 'frecuencia_20000'
]

# Crear un nuevo dato para la predicción
nuevo_dato = pd.DataFrame([[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]], columns=frecuencias)

# Usar el scaler cargado para transformar el nuevo dato
nuevo_dato_scaled = scaler_cargado.transform(nuevo_dato)

# Realizar la predicción con el modelo cargado
nueva_prediccion = modelo_cargado.predict(nuevo_dato_scaled)

# Mostrar el resultado
print(f"Edad estimada con el modelo cargado: {nueva_prediccion[0]:.1f} años")
