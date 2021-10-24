import pandas as pd


def load_dataset(filename):
    df = pd.read_csv(filename, header=None)
    num_columns = len(df.columns)
    df[num_columns - 1] = df[num_columns - 1].astype('category')
    df[num_columns - 1] = df[num_columns - 1].cat.codes
    return df.to_numpy()
