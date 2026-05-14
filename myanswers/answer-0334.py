import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_limpiar_datos(df=None):
    n_rows = random.randint(8, 15)
    n_cols = random.randint(3, 6)

    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]

    df_gen = pd.DataFrame(data, columns=cols)
    df_gen = pd.concat([df_gen, df_gen.iloc[:2]], ignore_index=True)

    for col in df_gen.columns:
        if random.random() > 0.5:
            df_gen.loc[df_gen.sample(frac=0.6).index, col] = np.nan

    input_data = {
        "df": df_gen.copy()
    }

    df_clean = df_gen.drop_duplicates()
    threshold = len(df_clean) * 0.5
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    output_data = df_clean

    return input_data, output_data


def limpiar_datos(df):
    # 1. Eliminar filas duplicadas
    df_clean = df.drop_duplicates()

    # 2. Eliminar columnas con más del 50% de valores nulos
    threshold = len(df_clean) * 0.5
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    return df_clean
