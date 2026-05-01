import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def predecir_produccion(df, modelo_tipo="linear", test_size=0.25):
    """
    Predice la producción diaria de oro (kg) a partir de variables operativas.

    Parámetros:
        df          : DataFrame con columnas operativas y 'produccion_oro_kg'.
        modelo_tipo : 'linear' para LinearRegression, 'ridge' para Ridge.
        test_size   : Fracción del dataset destinada a prueba (0.2 – 0.3).

    Retorna:
        metricas    : dict con MAE, RMSE y R².
        coeficientes: array con los coeficientes del modelo entrenado.
        modelo      : objeto del modelo ya entrenado.
    """
    # 1. Separar variables independientes y dependiente
    X = df.drop(columns=["produccion_oro_kg"])
    y = df["produccion_oro_kg"]

    # 2. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 3. Seleccionar modelo
    if modelo_tipo == "linear":
        modelo = LinearRegression()
    elif modelo_tipo == "ridge":
        modelo = Ridge(alpha=1.0)
    else:
        raise ValueError("Modelo no válido. Use 'linear' o 'ridge'")

    # 4. Entrenar
    modelo.fit(X_train, y_train)

    # 5. Predecir y evaluar
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metricas = {
        "MAE":  mae,
        "RMSE": rmse,
        "R2":   r2,
    }

    # 6. Retornar métricas, coeficientes y modelo
    return metricas, modelo.coef_, modelo
