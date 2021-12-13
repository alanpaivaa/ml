import numpy as np

np.random.seed(11)


def em(x, k):
    n, d = x.shape

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
        mk_t = mean.reshape(k, 1, d)
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

        mk_t = mean.reshape(k, 1, d)
        mk = mean.reshape(k, d, 1)
        cov = (resp.T.reshape(k, 1, n) * (xk - mk) @ (xk_t - mk_t)) / n_k.reshape(k, 1, 1)

        converged = error < epsilon
        # print(iterations, np.sum(error))

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

# plt.show()

