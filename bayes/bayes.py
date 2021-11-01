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
            cm.append(np.cov(xc, rowvar=False))
        self.cov_matrix = np.array(cm)

    def predict(self, x):
        d = x.shape[0]
        p_x = list()
        for c in range(self.num_classes):
            det_cov_matrix = np.linalg.det(self.cov_matrix[c])
            p = 1 / (np.power(2 * np.pi, d / 2) * np.sqrt(det_cov_matrix))
            p *= np.exp(-1 / 2 * ((x - self.means[c]).reshape((1, d)) @ np.linalg.inv(self.cov_matrix[c]) @ (x - self.means[c]).reshape(d, 1)).item())
            p *= self.priori[c]
            p_x.append(p)
        return np.argmax(np.array(p_x))
