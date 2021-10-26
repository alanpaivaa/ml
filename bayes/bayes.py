import numpy as np


class GaussianBayes:
    def __init__(self):
        self.mean = None
        self.priori = None
        self.cov_matrix = None

    # TODO: Make multivariate
    def train(self, dataset):
        # Get all classes
        classes = np.unique(dataset[:, -1])

        x = dataset[:, :-1]
        self.generate_cov_matrix(x)

        # self.mean = np.mean(dataset[0])
        # # Calculate priori for each class
        # s = np.repeat(np.array([classes]), dataset.shape[0], axis=0)  # Shape: (num_patterns, num_classes)
        # c = np.repeat(np.array([dataset[:, -1]]), classes.shape[0], axis=0).T  # Shape: (num_patterns< num_classes)
        # class_count = dataset.shape[0] - np.count_nonzero(c - s, axis=0)  # Shape: (1, num_classes)
        # self.priori = class_count / dataset.shape[0]  # Shape: (1, num_classes)

        print("Done training!")

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
        pass

