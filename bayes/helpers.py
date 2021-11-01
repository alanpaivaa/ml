import pandas as pd
import numpy as np


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


def univariate_gaussians(dataset, col):
    num_classes = int(np.max(dataset[:, -1]) + 1)
    gaussians = list()
    for c in range(num_classes):
        xc = dataset[dataset[:, -1] == c][:, col]
        mean = np.mean(xc)
        std = np.std(xc)
        min = np.min(xc)
        max = np.max(xc)
        offset = (max - min) * 0.4
        min -= offset
        max += offset
        x = np.linspace(min, max, 100)
        y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-1 * ((x - mean) ** 2) / (2 * (std ** 2)))
        gaussians.append((x, y))
    return np.array(gaussians)
