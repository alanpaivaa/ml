import numpy as np
import matplotlib.pyplot as plt
from bayes_mixture.helpers import load_dataset, write_dataset


class Dataset:
    def __init__(self, filename, encoding=None, features=None):
        self.filename = filename
        self.encoding = encoding
        self.features = features

    def load(self):
        dataset = load_dataset(self.filename)

        # Selecting only the needed columns
        if self.features is not None:
            dataset = dataset[:, self.features + [-1]]

        return dataset


def sample_points(num_samples, x_range, y_range, space_size):
    # Generate x and y possible values
    x_values = np.linspace(x_range[0], x_range[1], space_size).round(3)
    y_values = np.linspace(y_range[0], y_range[1], space_size).round(3)

    # Make coordinates as combination of x and y values
    xx, yy = np.meshgrid(x_values, y_values)
    xx = xx.reshape(1, -1)
    yy = yy.reshape(1, -1)

    grid = np.stack([xx.T, yy.T], axis=2).reshape(-1, 2)
    np.random.shuffle(grid)

    # Sample coordinates
    return grid[:num_samples]


def generate_artificial_dataset():
    num_samples = 25
    space_size = 1000
    k0 = sample_points(num_samples=num_samples, x_range=(0, 1), y_range=(2, 3), space_size=space_size)
    k1 = sample_points(num_samples=num_samples, x_range=(1, 2), y_range=(0, 1), space_size=space_size)
    k2 = sample_points(num_samples=num_samples, x_range=(1, 2), y_range=(1, 2), space_size=space_size)
    k3 = sample_points(num_samples=num_samples, x_range=(2, 3), y_range=(0, 1), space_size=space_size)
    k4 = sample_points(num_samples=num_samples, x_range=(2, 3), y_range=(1, 2), space_size=space_size)

    z = np.zeros((k0.shape[0], 1))
    k0 = np.hstack((k0, z))
    k1 = np.hstack((k1, z))
    k4 = np.hstack((k4, z))

    o = np.ones((k2.shape[0], 1))
    k2 = np.hstack((k2, o))
    k3 = np.hstack((k3, o))

    dataset = np.concatenate([k0, k1, k2, k3, k4])
    np.random.shuffle(dataset)

    c0 = dataset[dataset[:, -1] == 0]
    c1 = dataset[dataset[:, -1] == 1]
    plt.scatter(c0[:, 0], c0[:, 1])
    plt.scatter(c1[:, 0], c1[:, 1])
    plt.show()

    write_dataset(dataset, "bayes_mixture/datasets/artificial.csv")
