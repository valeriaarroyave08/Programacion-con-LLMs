import numpy as np
import pandas as pd


def filtro_ema_adaptativo(df, valor_col, ventana_volatilidad=20):
    """
    Aplica un filtro EMA adaptativo basado en la volatilidad local.

    Parámetros:
        df                  : DataFrame con la señal del sensor.
        valor_col           : Nombre de la columna con los valores.
        ventana_volatilidad : Ventana para calcular la desviación estándar móvil.

    Retorna:
        DataFrame original con las columnas 'senal_limpia' y 'alpha_dinamico'.
    """
    df_res = df.copy()
    serie = df_res[valor_col].values

    # 1. Volatilidad local (std móvil) con Pandas
    volatilidad = (
        df_res[valor_col]
        .rolling(window=ventana_volatilidad, min_periods=1)
        .std()
        .fillna(0)
        .values
    )

    # 2. Alpha dinámico normalizado en [0.1, 0.9]
    max_vol = np.max(volatilidad) if np.max(volatilidad) > 0 else 1
    alphas = 0.1 + 0.8 * (volatilidad / max_vol)

    # 3. Aplicar EMA adaptativo
    ema_filtrada = np.zeros(len(serie))
    ema_filtrada[0] = serie[0]

    for i in range(1, len(serie)):
        a = alphas[i]
        ema_filtrada[i] = a * serie[i] + (1 - a) * ema_filtrada[i - 1]

    df_res["senal_limpia"]   = ema_filtrada
    df_res["alpha_dinamico"] = alphas

    return df_res
