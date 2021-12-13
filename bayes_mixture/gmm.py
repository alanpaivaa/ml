import numpy as np


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def em(x, k, log_error=False):
    n, d = x.shape

    pi = np.ones(k) / k  # Initial pi: priori for k
    mean = x[np.random.choice(np.arange(0, x.shape[0]), k)]
    cov = np.cov(x, rowvar=False)  # Initial means from random points
    cov = np.repeat(np.array([cov]), k, axis=0)  # Initial cov matrices repeated x.cov

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

        # Update covariance matrices
        mk_t = mean.reshape(k, 1, d)
        mk = mean.reshape(k, d, 1)
        cov = (resp.T.reshape(k, 1, n) * (xk - mk) @ (xk_t - mk_t)) / n_k.reshape(k, 1, 1)

        converged = error < epsilon

        if log_error:
            print(iterations, np.sum(error))

    return iterations, pi, mean, cov
