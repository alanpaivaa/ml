import numpy as np


class GaussianBayes:
    def __init__(self):
        self.num_classes = None
        self.means = None
        self.cov_matrix = None
        self.priori = None

    def train(self, dataset):
        self.num_classes = np.max(dataset[:, -1]).astype(int) + 1
        self.generate_priori(dataset)
        self.generate_means(dataset)
        self.generate_cov_matrix(dataset)

    def generate_priori(self, dataset):
        p = list()
        for c in range(self.num_classes):
            s = np.sum(dataset[:, -1] == c)
            p.append(s / dataset.shape[0])
        self.priori = np.array(p)

    def generate_means(self, dataset):
        m = list()
        for c in range(self.num_classes):
            points = dataset[dataset[:, -1] == c][:, :-1]
            mean = np.mean(points, axis=0)
            m.append(mean)
        self.means = np.array(m)

    def generate_cov_matrix(self, dataset):
        cm = list()
        for c in range(self.num_classes):
            xc = dataset[dataset[:, -1] == c][:, :-1]
            cm.append(np.cov(xc, rowvar=False))  # Quadratic
        self.cov_matrix = np.array(cm)

    def predict(self, x_t):
        g_x = list()
        for c in range(self.num_classes):
            cov_inv = np.linalg.inv(self.cov_matrix[c])
            g_i = -0.5 * (x_t @ cov_inv @ x_t.T)
            g_i += 0.5 * (x_t @ cov_inv @ self.means[c].T)
            g_i -= 0.5 * (self.means[c] @ cov_inv @ self.means[c].T)
            g_i += 0.5 * (self.means[c] @ cov_inv @ x_t)
            g_i += np.log(self.priori[c])
            g_i -= 0.5 * np.log(2 * np.pi)
            g_i -= 0.5 * np.log(np.linalg.det(self.cov_matrix[c]))
            g_x.append(g_i)
        return np.argmax(np.array(g_x))

