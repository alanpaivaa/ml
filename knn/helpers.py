from csv import reader as csv_reader
import random
import math
from functools import reduce


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv_reader(file)
        for row in reader:
            dataset.append(row)
    return dataset


def load_dataset(filename, last_column):
    dataset = load_csv(filename)
    offset = 1
    if last_column:
        offset = 0
    for row in dataset:
        for i in range(len(row) - offset):
            # Best effort: we just skip in case we can't convert a row to number
            try:
                row[i] = float(row[i])
            except ValueError:
                row[i] = 0
    return dataset


def train_test_split(dataset, ratio=0.8, shuffle=False):
    dataset_copy = dataset.copy()
    if shuffle:
        random.shuffle(dataset_copy)
    train_index = int(len(dataset_copy) * ratio)
    training_set = dataset_copy[:train_index]
    test_set = dataset_copy[train_index:]
    return training_set, test_set


def mean(vector):
    return reduce(lambda x, y: x + y, vector) / len(vector)


def standard_deviation(vector):
    # Avoid zero division
    if len(vector) == 1:
        return 0
    m = mean(vector)
    mean_difference = map(lambda x: (x - m) ** 2, vector)
    return math.sqrt(reduce(lambda x, y: x + y, mean_difference) / (len(vector) - 1))