import pandas as pd


def limpiar_datos(df=generar_caso_de_uso_limpiar_datos()[0]):
    """
    Limpia un DataFrame eliminando duplicados y columnas con >50% nulos.

    Parámetros:
        df : DataFrame original.

    Retorna:
        DataFrame limpio.
    """
    # 1. Eliminar filas duplicadas
    df_clean = df.drop_duplicates()

    # 2. Eliminar columnas con más del 50% de valores nulos
    threshold = len(df_clean) * 0.5
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    return df_clean
