import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import random


def generar_caso_de_uso_optimizar_modelo_logistica():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función optimizar_modelo_logistica.
    """

    # ---------------------------------------------------------
    # 1. Generar dimensiones aleatorias del dataset
    # ---------------------------------------------------------
    n_rows = random.randint(40, 100)

    # ---------------------------------------------------------
    # 2. Crear variables simulando un problema logístico
    # ---------------------------------------------------------
    X = pd.DataFrame({
        "distancia_km": np.random.uniform(50, 5000, n_rows),
        "peso_toneladas": np.random.uniform(1, 40, n_rows),
        "numero_aduanas": np.random.randint(1, 6, n_rows)
    })

    # Generamos el tiempo de entrega con algo de ruido
    y = (
        0.04 * X["distancia_km"] +
        2 * X["peso_toneladas"] +
        8 * X["numero_aduanas"] +
        np.random.normal(0, 30, n_rows)
    )

    # ---------------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular OUTPUT esperado replicando la lógica
    # ---------------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("randomforest", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "randomforest__n_estimators": [50, 100],
        "randomforest__max_depth": [None, 5, 10]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X, y)

    output_data = {
        "mejor_modelo": grid.best_estimator_,
        "mejor_mae": abs(grid.best_score_)
    }

    return input_data, output_data
