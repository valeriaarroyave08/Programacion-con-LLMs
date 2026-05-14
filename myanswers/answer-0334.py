import pandas as pd
import numpy as np
import random

def limpiar_datos(df):
    # 1. Eliminar filas duplicadas
    df_clean = df.drop_duplicates()

    # 2. Eliminar columnas con más del 50% de valores nulos
    threshold = len(df_clean) * 0.5
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    return df_clean
