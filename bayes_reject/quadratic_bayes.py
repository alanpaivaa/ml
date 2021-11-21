import numpy as np


class QuadraticBayes:
    def __init__(self, wr):
        self.wr = wr
        self.num_classes = None
        self.means = None
        self.cov_matrix = None
        self.priori = None
        self.t = None

    def train(self, dataset):
        self.num_classes = np.max(dataset[:, -1]).astype(int) + 1
        self.generate_priori(dataset)
        self.generate_means(dataset)
        self.generate_cov_matrix(dataset)
        self.minimize_rejection_error(dataset)

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

    def cov_matrix_for_class(self, c):
        return self.cov_matrix[c]

    def generate_cov_matrix(self, dataset):
        cm = list()
        for c in range(self.num_classes):
            xc = dataset[dataset[:, -1] == c][:, :-1]
            cm.append(np.cov(xc, rowvar=False))
        self.cov_matrix = np.array(cm)

    def minimize_rejection_error(self, dataset):
        ts = np.linspace(0, 0.5, 50)
        metrics = []
        for t in ts:
            metrics.append(self.metrics_for_t(dataset, t))
        metrics = np.array(metrics)
        self.t = ts[np.argmin(metrics)]

    def metrics_for_t(self, dataset, t):
        rejected = 0.0
        errors = 0.0
        for x in dataset:
            predicted = self.predict_t(x[:-1], t)
            if predicted < 0:
                rejected += 1
            elif predicted != x[-1]:
                errors += 1
        n = dataset.shape[0]
        if rejected < n:
            e_t = errors / (n - rejected)
        else:
            e_t = 0
        r_t = rejected / n
        return e_t + self.wr * r_t

    def predict(self, x_t):
        return self.predict_t(x_t, self.t)

    def predict_t(self, x_t, t):
        p_x = list()
        d = x_t.shape[0]
        x_t = x_t.reshape((-1, d))
        x = x_t.T

        for c in range(self.num_classes):
            cov = self.cov_matrix_for_class(c)
            cov_inv = np.linalg.inv(cov)
            u = self.means[c].reshape((-1, d)).T
            p_i = np.exp((-0.5 * (x - u).T @ cov_inv @ (x - u)).item())
            p_i /= (2 * np.pi) ** (x_t.shape[0]/2) * np.linalg.det(cov) ** 0.5
            p_i *= self.priori[c]
            p_x.append(p_i)

        p_x = np.array(p_x)
        p_x /= p_x.sum()

        p_x -= t
        mi = np.argmax(np.array(p_x))

        if p_x[mi] < 0.5:
            return -1
        return mi
