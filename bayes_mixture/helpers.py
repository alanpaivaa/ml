import numpy as np
import pandas as pd


def load_dataset(filename):
    df = pd.read_csv(filename, header=None)
    num_columns = len(df.columns)
    df[num_columns - 1] = df[num_columns - 1].astype('category')
    df[num_columns - 1] = df[num_columns - 1].cat.codes
    return df.to_numpy()


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        np.random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set
