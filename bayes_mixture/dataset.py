from bayes_mixture.helpers import load_dataset
import random
import numpy as np


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
    x_values = np.linspace(x_range[0], x_range[1], space_size)
    y_values = np.linspace(y_range[0], y_range[1], space_size)

    # Make coordinates as combination of x and y values
    coordinates = [[round(x, 2), round(y, 2)] for x in x_values for y in y_values]

    # Sample coordinates
    return random.choices(coordinates, k=num_samples)
