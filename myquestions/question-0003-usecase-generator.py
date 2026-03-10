import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import random


def generar_caso_de_uso_seleccionar_mejores_variables():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función seleccionar_mejores_variables.
    """

    # ---------------------------------------------------------
    # 1. Definir dimensiones aleatorias
    # ---------------------------------------------------------
    n_rows = random.randint(60, 120)
    n_features = random.randint(8, 12)

    columnas = [f"variable_{i}" for i in range(n_features)]

    # ---------------------------------------------------------
    # 2. Crear dataset financiero simulado
    # ---------------------------------------------------------
    X = pd.DataFrame(
        np.random.randn(n_rows, n_features),
        columns=columnas
    )

    y = np.random.randint(0, 2, n_rows)

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

    selector = SelectKBest(score_func=f_classif, k=5)

    X_new = selector.fit_transform(X, y)

    mask = selector.get_support()

    variables = list(X.columns[mask])

    X_reducido = pd.DataFrame(X_new, columns=variables)

    output_data = {
        "variables_seleccionadas": variables,
        "X_reducido": X_reducido
    }

    return input_data, output_data
