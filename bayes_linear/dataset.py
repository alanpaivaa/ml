from bayes_linear.helpers import load_dataset, write_dataset
import random
import matplotlib.pyplot as plt
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


def generate_artificial_dataset(plotting_enabled):
    num_samples = 50
    space_size = 1000
    circles = sample_points(num_samples=num_samples, x_range=(0, 5), y_range=(5.5, 10.5), space_size=space_size)
    stars = sample_points(num_samples=num_samples, x_range=(5.5, 10.5), y_range=(0, 5), space_size=space_size)
    triangles = sample_points(num_samples=num_samples, x_range=(11, 16), y_range=(5.5, 10.5), space_size=space_size)

    # Plot points
    if plotting_enabled:
        circles_x = [point[0] for point in circles]
        circles_y = [point[1] for point in circles]
        plt.scatter(circles_x, circles_y, marker="o")

        stars_x = [point[0] for point in stars]
        stars_y = [point[1] for point in stars]
        plt.scatter(stars_x, stars_y, marker="*")

        triangles_x = [point[0] for point in triangles]
        triangles_y = [point[1] for point in triangles]
        plt.scatter(triangles_x, triangles_y, marker="^")

        plt.show()

    dataset = [point + [0] for point in circles] + \
              [point + [1] for point in stars] + \
              [point + [2] for point in triangles]

    random.shuffle(dataset)

    write_dataset(np.array(dataset), "bayes_linear/datasets/artificial.csv")
