import numpy as np


class GaussianBayes:
    def __init__(self):
        self.all_classes = None
        self.means = None
        self.cov_matrix = None

    def train(self, dataset):
        self.all_classes = np.unique(dataset[:, -1]).astype(int)
        self.generate_means(dataset)
        self.generate_cov_matrix(dataset[:, :-1])

    def generate_means(self, dataset):
        self.means = np.array([])
        for c in self.all_classes:
            points = dataset[dataset[:, -1] == c][:, :-1]
            mean = np.mean(points, axis=0)
            self.means = np.append(self.means, mean).reshape((-1, points.shape[1]))

    def generate_cov_matrix(self, x):
        num_patterns = x.shape[0]
        num_attributes = x.shape[1]
        means = np.mean(x, axis=0)
        self.cov_matrix = np.zeros((num_attributes, num_attributes))
        for i in range(0, num_attributes):
            for j in range(0, num_attributes):
                x_i, x_j = x[:, i], x[:, j]
                self.cov_matrix[i][j] = np.sum((x_i - means[i]) * (x_j - means[j])) / (num_patterns - 1)

        # Asserting correctness of the generated covariance matrix with the built in numpy implementation
        assert np.unique(np.isclose(self.cov_matrix, np.cov(x.T))).item()

    def predict(self, x):
        d = x.shape[0]
        p_x = np.array([])
        det_cov_matrix = np.linalg.det(self.cov_matrix)
        for c in self.all_classes:
            p = 1 / (np.power(2 * np.pi, d / 2) * np.power(det_cov_matrix, d / 2))
            p *= np.exp(-0.5 * ((x - self.means[c]).reshape((1, d)) @ np.linalg.inv(self.cov_matrix) @ (x - self.means[c]).reshape(d, 1)).item())
            p_x = np.append(p_x, p)
        return np.argmax(p_x)

