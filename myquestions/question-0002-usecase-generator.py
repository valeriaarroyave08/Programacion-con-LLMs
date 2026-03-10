import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import random


def generar_caso_de_uso_clasificar_churn_svm():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función clasificar_churn_svm.
    """

    # ---------------------------------------------------------
    # 1. Generar tamaño aleatorio del dataset
    # ---------------------------------------------------------
    n_rows = random.randint(60, 120)

    # ---------------------------------------------------------
    # 2. Crear variables simulando clientes de telecom
    # ---------------------------------------------------------
    X = pd.DataFrame({
        "duracion_contrato": np.random.randint(1, 36, n_rows),
        "llamadas_soporte": np.random.randint(0, 20, n_rows),
        "tipo_plan": np.random.randint(1, 4, n_rows)
    })

    # Variable objetivo: churn
    y = np.random.randint(0, 2, n_rows)

    # ---------------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular OUTPUT esperado
    # ---------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )

    modelo = SVC(kernel="rbf")

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    output_data = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred, zero_division=0)
    }

    return input_data, output_data
