import numpy as np
from bayes_linear.quadratic_bayes import QuadraticBayes

LINEAR_AGGREGATION_NAIVE = 'naive'
LINEAR_AGGREGATION_POOL = 'pool'


class LinearBayes(QuadraticBayes):
    def __init__(self, aggregation=LINEAR_AGGREGATION_NAIVE):
        super().__init__()
        self.aggregation = aggregation

    def generate_cov_matrix(self, dataset):
        if self.aggregation == LINEAR_AGGREGATION_NAIVE:
            x = dataset[:, :-1]
            self.cov_matrix = np.cov(x, rowvar=False)
        elif self.aggregation == LINEAR_AGGREGATION_POOL:
            super().generate_cov_matrix(dataset)
            self.cov_matrix *= self.priori.reshape(-1, 1, 1)
            self.cov_matrix = self.cov_matrix.sum(axis=0)
        else:
            raise Exception("Invalid aggregation")

    def cov_matrix_for_class(self, c):
        return self.cov_matrix
