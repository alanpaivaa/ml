import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)

x = np.linspace(0.1, 0.5, 10)
x = np.concatenate((x, np.linspace(1.1, 1.5, 10)))
x = np.concatenate((x, np.linspace(2.1, 2.5, 10)))

n = x.shape[0]
k = 3

pi = np.ones(k) / k
mean = np.random.sample(k)
variance = np.repeat(x.var(), k)

epsilon = 1e-10
converged = False
iterations = 0
while not converged:
    iterations += 1

    # Calculate responsibilities
    resp = np.repeat(np.expand_dims(x, axis=0), k, axis=0).T
    resp = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-1 * ((resp - mean) ** 2) / (2 * variance))
    resp *= pi
    resp /= np.repeat(np.expand_dims(resp.sum(axis=1), axis=0), k, axis=0).T

    n_k = resp.sum(axis=0)

    new_mean = np.sum(resp * np.repeat(np.expand_dims(x, axis=0), k, axis=0).T, axis=0) / n_k
    error = np.sum(np.abs(new_mean - mean))
    mean = new_mean

    variance = np.sum(resp * ((np.repeat(np.expand_dims(x, axis=0), k, axis=0).T - mean) ** 2), axis=0) / n_k

    # New pi
    pi = n_k / n

    converged = error <= epsilon

print("Iterations:", iterations)


# print(mean)

plt.scatter(x, np.zeros(x.shape[0]))


s = np.linspace(0.1, 2.5)
for i in range(k):
    gaussian = (1 / (np.sqrt(2 * np.pi * variance[i]))) * np.exp(-1 * ((s - mean[i]) ** 2) / (2 * variance[i]))
    plt.plot(s, gaussian)

plt.show()
