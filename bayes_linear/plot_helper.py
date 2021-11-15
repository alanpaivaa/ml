import matplotlib.pyplot as plt
import numpy as np
from bayes_linear.helpers import univariate_gaussians


def plot_decision_surface(model, dataset, title=None, xlabel=None, ylabel=None, legend=None):
    offset_percentage = 0.06

    x_min = np.min(dataset[:, 0])
    x_max = np.max(dataset[:, 0])
    offset_x = (x_max - x_min) * offset_percentage
    x_values = np.linspace(x_min - offset_x, x_max + offset_x, 125)

    y_min = np.min(dataset[:, 1])
    y_max = np.max(dataset[:, 1])
    offset_y = (y_max - y_min) * offset_percentage
    y_values = np.linspace(y_min - offset_y, y_max + offset_y, 125)

    grid = np.array([[x, y] for x in x_values for y in y_values])
    predictions = np.array([model.predict(row) for row in grid])

    classes = np.unique(predictions)
    colors = ["cornflowerblue", "forestgreen", "purple", "brown", "yellow", "orange"]
    markers = ["o", "*", "^", "s", "p", "h"]
    for c in classes:
        points = grid[predictions == c]
        plt.scatter(points[:, 0], points[:, 1], marker="s", color=colors[c], alpha=0.05)
        set_points = dataset[dataset[:, -1] == c]
        label = None
        if legend is not None:
            label = legend[c]
        plt.scatter(set_points[:, 0], set_points[:, 1], color=colors[c], marker=markers[c], label=label)

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


def plot_pdf(dataset, col, title=None, x_label=None, y_label=None, legend=None):
    num_classes = int(np.max(dataset[:, -1]) + 1)
    gaussians = univariate_gaussians(dataset, col)
    colors = ["cornflowerblue", "forestgreen", "purple"]
    for c in range(num_classes):
        xc = dataset[dataset[:, -1] == c][:, col]
        plt.hist(xc, density=True, color=colors[c], alpha=0.25, stacked=True)
        label = None
        if legend is not None:
            label = legend[c]
        plt.plot(gaussians[c][0], gaussians[c][1], color=colors[c], label=label)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if legend is not None:
        plt.legend()
    plt.show()
