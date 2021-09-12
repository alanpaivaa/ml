import matplotlib.pyplot as plt
import numpy as np


def plot_decision_surface(model, dataset, title=None, xlabel=None, ylabel=None, legend=None):
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    grid = np.array([[x, y] for x in x_values for y in y_values])
    predictions = np.array([model.predict(list(row)) for row in grid])

    np_dataset = np.array(dataset)

    classes = np.unique(predictions)
    colors = ["cornflowerblue", "forestgreen", "salmon"]
    markers = ["o", "*", "^"]
    for c in classes:
        points = grid[predictions == c]
        plt.scatter(points[:, 0], points[:, 1], marker="s", color=colors[c], alpha=0.05)
        set_points = np_dataset[np_dataset[:, -1] == c]
        plt.scatter(set_points[:, 0], set_points[:, 1], color=colors[c], marker=markers[c])

    plt.margins(x=0, y=0)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if legend is not None:
        plt.legend()

    plt.show()
