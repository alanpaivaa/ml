import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bayes_mixture.dataset import Dataset
from bayes_mixture.normalizer import Normalizer


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
    num_samples = 50
    space_size = 1000
    k0 = sample_points(num_samples=num_samples, x_range=(0, 1), y_range=(1, 2), space_size=space_size)
    k1 = sample_points(num_samples=num_samples, x_range=(1, 2), y_range=(0, 1), space_size=space_size)

    dataset = np.concatenate([k0, k1])
    np.random.shuffle(dataset)

    # plt.scatter(dataset[:, 0], dataset[:, 1])
    # plt.show()

    return dataset

    # dataset = [point + [0] for point in circles] + \
    #           [point + [1] for point in stars] + \
    #           [point + [2] for point in triangles]
    #
    # np.random.shuffle(dataset)

    # write_dataset(dataset, "assignment3/datasets/artificial.csv")


def plot_artificial_dataset(is_3d=False):
    dataset = generate_artificial_dataset()
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


np.random.seed(11)

# plot_artificial_dataset(is_3d=True)
# exit(0)

# x = np.linspace(0.1, 0.5, 10)
# x = np.concatenate((x, np.linspace(1.1, 1.5, 10)))
# x = np.concatenate((x, np.linspace(2.1, 2.5, 10)))

x = Dataset("datasets/dermatology.csv").load()
normalizer = Normalizer()
normalizer.fit(x[:, :-1])
normalized_dataset = [np.append(normalizer.normalize(row[:-1]), row[-1]) for row in x]
x = np.array(normalized_dataset)
x = x[:, :-1]
# x = generate_artificial_dataset()

n, d = x.shape
k = 2

pi = np.ones(k) / k  # Initial pi: priori for k
mean = x[np.random.choice(np.arange(0, x.shape[0]), k)]  # Initial means from random points
cov = np.repeat(np.array([np.cov(x, rowvar=False)]), k, axis=0)  # Initial cov matrices repeated x.cov

epsilon = 1e-8
converged = False
iterations = 0


while not converged:
    iterations += 1

    # Calculate responsibilities
    xk_t = np.repeat(np.array([x]), k, axis=0)
    mk_t = mn = mean.reshape(k, 1, d)
    xk = np.repeat(np.array([x.T]), k, axis=0)
    mk = mean.reshape(k, d, 1)
    resp = np.exp(-0.5 * np.diagonal((xk_t - mk_t) @ np.linalg.inv(cov) @ (xk - mk), axis1=1, axis2=2).T)
    resp /= ((2 * np.pi) ** (d / 2)) * np.sqrt(np.linalg.det(cov))
    resp *= pi
    resp /= resp.sum(axis=1).reshape(n, 1)

    n_k = resp.sum(axis=0)

    # Update Pi vector
    pi = n_k / n

    # Update mean
    new_mean = np.sum(resp.T.reshape(k, n, 1) * xk_t, axis=1) / n_k.reshape(k, 1)
    error = np.sum(np.abs(new_mean - mean))
    mean = new_mean

    # new_cov = list()
    # for p in range(k):
    #     cin = np.zeros((d, d))
    #     for b in range(n):
    #         cin += resp[b, p] * (x[b].reshape(d, 1) - mean[p].reshape(d, 1)) @ (x[b].reshape(1, d) - mean[p].reshape(1, d))
    #     cin /= n_k[p]
    #     new_cov.append(cin)
    # cov = np.array(new_cov)

    mk_t = mean.reshape(k, 1, d)
    mk = mean.reshape(k, d, 1)
    cov = (resp.T.reshape(k, 1, n) * (xk - mk) @ (xk_t - mk_t)) / n_k.reshape(k, 1, 1)

    converged = error < epsilon
    print(iterations, np.sum(error))

print("Iterations:", iterations)

# k = 0
# dataset = x
#
# grid_x = np.linspace(-0.5, 2.5, 100)
# grid_y = np.linspace(-0.5, 2.5, 100)
# xx, yy = np.meshgrid(grid_x, grid_y)
#
# # 3D
# xx = xx.reshape(1, -1)
# yy = yy.reshape(1, -1)
# grid = np.stack([xx.T, yy.T], axis=2).reshape(-1, 2)
# z = list()
# for x_t in grid:
#     res0 = np.exp(-0.5 * (x_t - mean[0]) @ np.linalg.inv(cov[0]) @ (x_t - mean[0]).T)
#     res0 /= (2 * np.pi) * np.sqrt(np.linalg.det(cov[0]))
#
#     # K1
#     res1 = np.exp(-0.5 * (x_t - mean[1]) @ np.linalg.inv(cov[1]) @ (x_t - mean[1]).T)
#     res1 /= (2 * np.pi) * np.sqrt(np.linalg.det(cov[1]))
#
#     res = pi[0] * res0 + pi[1] * res1
#     z.append(res)
# z = np.array(z)
# mpl.use('macosx')
# ax = plt.axes(projection="3d")
# ax.scatter3D(dataset[:, 0], dataset[:, 1], np.zeros(dataset.shape[0]), color='white')
# ax.plot3D(grid[:, 0], grid[:, 1], z)

# Contour
# z = list()
# for x, y in zip(xx, yy):
#     row = list()
#     for i in range(len(x)):
#         x_t = np.array([y[i], x[i]])
#         # K0
#         res0 = np.exp(-0.5 * (x_t - mean[0]) @ np.linalg.inv(cov[0]) @ (x_t - mean[0]).T)
#         res0 /= (2 * np.pi) * np.sqrt(np.linalg.det(cov[0]))
#
#         # K1
#         res1 = np.exp(-0.5 * (x_t - mean[1]) @ np.linalg.inv(cov[1]) @ (x_t - mean[1]).T)
#         res1 /= (2 * np.pi) * np.sqrt(np.linalg.det(cov[1]))
#
#         res = pi[0] * res0 + pi[1] * res1
#
#         row.append(res)
#     z.append(row)
# z = np.array(z)
# fig, ax = plt.subplots(1, 1)
# cp = ax.contourf(xx, yy, z)
# fig.colorbar(cp)
# ax.scatter(dataset[:, 0], dataset[:, 1], color='white')

plt.show()
