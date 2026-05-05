import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_datos():

    n_rows = random.randint(8, 15)
    n_cols = random.randint(3, 6)

    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]

    df = pd.DataFrame(data, columns=cols)

    df = pd.concat([df, df.iloc[:2]], ignore_index=True)

    for col in df.columns:
        if random.random() > 0.5:
            df.loc[df.sample(frac=0.6).index, col] = np.nan

    input_data = {
        "df": df.copy()
    }

    df_clean = df.drop_duplicates()

    threshold = len(df_clean) * 0.5
    df_clean = df_clean.dropna(thresh=threshold, axis=1)

    output_data = df_clean

    return input_data, output_data
