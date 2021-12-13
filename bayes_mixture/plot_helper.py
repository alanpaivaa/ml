import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_artificial_dataset(dataset, is_3d=False):
    mean_t = dataset.mean(axis=0)
    cov = np.cov(dataset, rowvar=False)

    grid_x = np.linspace(-0.5, 2.5, 100)
    grid_y = np.linspace(-0.5, 2.5, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)

    if is_3d:  # 3D
        xx = xx.reshape(1, -1)
        yy = yy.reshape(1, -1)
        grid = np.stack([xx.T, yy.T], axis=2).reshape(-1, 2)
        z = list()
        for x_t in grid:
            res = np.exp(-0.5 * (x_t - mean_t) @ np.linalg.inv(cov) @ (x_t - mean_t).T)
            res /= (2 * np.pi) * np.sqrt(np.linalg.det(cov))
            z.append(res)
        z = np.array(z)
        mpl.use('macosx')
        ax = plt.axes(projection="3d")
        ax.scatter3D(dataset[:, 0], dataset[:, 1], np.zeros(dataset.shape[0]))
        ax.plot3D(grid[:, 0], grid[:, 1], z)

    else:  # Contour
        z = list()
        for x, y in zip(xx, yy):
            row = list()
            for i in range(len(x)):
                x_t = np.array([y[i], x[i]])
                res = np.exp(-0.5 * (x_t - mean_t) @ np.linalg.inv(cov) @ (x_t - mean_t).T)
                res /= (2 * np.pi) * np.sqrt(np.linalg.det(cov))
                row.append(res)
            z.append(row)
        z = np.array(z)
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(xx, yy, z)
        fig.colorbar(cp)

    plt.show()
