from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


def optimizar_modelo_bioinformatica(X, y):
    """
    Optimiza un modelo ElasticNet para predicción de niveles proteicos
    a partir de expresión génica, usando GridSearchCV con cv=3.

    Parámetros:
        X : array-like, matriz de features (genes).
        y : array-like, variable objetivo (nivel de proteína).

    Retorna:
        mejor_r2      : float, mejor R² redondeado a 6 decimales.
        mejores_params: dict, mejores hiperparámetros encontrados.
    """
    model = ElasticNet()

    param_grid = {
        'alpha':    [0.1, 1.0, 10.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
    grid_search.fit(X, y)

    mejor_r2       = round(grid_search.best_score_, 6)
    mejores_params = grid_search.best_params_

    return mejor_r2, mejores_params
