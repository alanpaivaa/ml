import numpy as np
# import matplotlib.pyplot as plt
from bayes_reject.helpers import load_dataset
from bayes_reject.helpers import write_dataset


class Dataset:
    def __init__(self, filename, encoding=None, features=None, klass=None):
        self.filename = filename
        self.encoding = encoding
        self.features = features
        self.klass = klass

    def load(self):
        dataset = load_dataset(self.filename)

        # Selecting only the needed columns
        if self.features is not None:
            dataset = dataset[:, self.features + [-1]]

        if self.klass is not None:
            in_indexes = dataset[:, -1] == self.klass
            out_indexes = dataset[:, -1] != self.klass
            dataset[in_indexes, -1] = [0]
            dataset[out_indexes, -1] = [1]

        return dataset


def generate_artificial_dataset():
    x = np.linspace(0, 5, 50)

    x1 = x + np.random.uniform(low=0.1, high=0.5, size=(50,))
    c1 = -2 * x1 - np.random.uniform(low=-2, high=10, size=(50,))

    x2 = x + np.random.uniform(low=0.1, high=0.5, size=(50,))
    c2 = -2 * x2 + np.random.uniform(low=-2, high=10, size=(50,))

    f1 = np.stack((x1, c1, np.zeros(x1.shape)), axis=1)
    f2 = np.stack((x2, c2, np.ones(x2.shape)), axis=1)
    dataset = np.concatenate((f1, f2))
    np.random.shuffle(dataset)
    dataset = np.round(dataset, 2)

    # p0 = dataset[dataset[:, -1] == 0]
    # plt.scatter(p0[:, 0], p0[:, 1])
    #
    # p1 = dataset[dataset[:, -1] == 1]
    # plt.scatter(p1[:, 0], p1[:, 1])

    write_dataset(dataset, 'bayes_reject/datasets/artificial.csv')

    # plt.show()
