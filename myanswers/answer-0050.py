import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predecir_produccion(df, modelo_tipo="linear", test_size=0.25):
    X = df.drop(columns=["produccion_oro_kg"])
    y = df["produccion_oro_kg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if modelo_tipo == "linear":
        modelo = LinearRegression()
    elif modelo_tipo == "ridge":
        modelo = Ridge(alpha=1.0)
    else:
        raise ValueError("Modelo no válido. Use 'linear' o 'ridge'")

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metricas = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return metricas, modelo.coef_, modelo
