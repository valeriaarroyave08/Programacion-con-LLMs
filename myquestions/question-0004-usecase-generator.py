import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import random


def generar_caso_de_uso_detectar_anomalias_transacciones():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función detectar_anomalias_transacciones.
    """

    # ---------------------------------------------------------
    # 1. Generar número aleatorio de transacciones
    # ---------------------------------------------------------
    n_rows = random.randint(80, 150)

    # ---------------------------------------------------------
    # 2. Crear dataset de transacciones simuladas
    # ---------------------------------------------------------
    X = pd.DataFrame({
        "monto": np.random.uniform(5, 500, n_rows),
        "hora": np.random.randint(0, 24, n_rows),
        "frecuencia_usuario": np.random.randint(1, 50, n_rows),
        "distancia_km": np.random.uniform(0, 100, n_rows)
    })

    # ---------------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular OUTPUT esperado
    # ---------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    modelo.fit(X_scaled)

    predicciones = modelo.predict(X_scaled)

    output_data = predicciones

    return input_data, output_data
