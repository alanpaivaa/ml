import numpy as np
from bayes_linear.quadratic_bayes import QuadraticBayes


class LinearBayes(QuadraticBayes):
    def generate_cov_matrix(self, dataset):
        x = dataset[:, :-1]
        self.cov_matrix = np.cov(x, rowvar=False)

    def cov_matrix_for_class(self, c):
        return self.cov_matrix
