import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
data = {
    'edad': [30, 65, 25, 70, 40, 55, 35, 50, 28, 60, 45, 38],
    'frecuencia_250': [15, 30, 10, 45, 20, 25, 18, 28, 12, 32, 22, 19],
    'frecuencia_1000': [20, 40, 15, 55, 25, 35, 22, 38, 17, 45, 28, 24],
    'frecuencia_4000': [25, 60, 18, 70, 30, 50, 28, 55, 20, 65, 35, 30],
    'frecuencia_9000': [30, 65, 20, 75, 35, 55, 32, 60, 25, 70, 40, 35],
    'frecuencia_10000': [35, 70, 25, 80, 40, 60, 38, 65, 30, 75, 45, 40],
    'frecuencia_15000': [40, 75, 30, 85, 45, 65, 45, 70, 35, 80, 50, 45],
    'frecuencia_16000': [45, 80, 35, 90, 50, 70, 50, 75, 40, 85, 55, 50],
    'frecuencia_17000': [50, 85, 40, 95, 55, 75, 55, 80, 45, 90, 60, 55],
    'frecuencia_18000': [55, 90, 45, 100, 60, 80, 60, 85, 50, 95, 65, 60],
    'frecuencia_19000': [60, 95, 50, 105, 65, 85, 65, 90, 55, 100, 70, 65],
    'frecuencia_20000': [65, 100, 55, 110, 70, 90, 70, 95, 60, 105, 75, 70]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Separar las características (X) y la variable objetivo (y)
frecuencias = [
    'frecuencia_250', 'frecuencia_1000', 'frecuencia_4000',
    'frecuencia_9000', 'frecuencia_10000', 'frecuencia_15000',
    'frecuencia_16000', 'frecuencia_17000', 'frecuencia_18000',
    'frecuencia_19000', 'frecuencia_20000'
]

X = df[frecuencias]
y = df['edad']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajustar el escalador y transformar los datos de entrenamiento

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)

# Guardar el modelo de regresión lineal y el scaler entrenado
joblib.dump(modelo, 'modelo_lineal_entrenado.pkl')  # Guardar el modelo de regresión lineal
joblib.dump(scaler, 'scaler_entrenado.pkl')  # Guardar el escalador

print("El modelo y el escalador han sido guardados exitosamente.")
