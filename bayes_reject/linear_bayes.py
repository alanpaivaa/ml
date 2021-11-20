import numpy as np
from bayes_reject.quadratic_bayes import QuadraticBayes

AGGREGATION_NAIVE = 'naive'
AGGREGATION_POOL = 'pool'
AGGREGATION_DIAGONAL_VARIANCE = 'diagonal_variance'
AGGREGATION_DIAGONAL_EQUAL_PRIORI = 'diagonal_equal_priori'


class LinearBayes(QuadraticBayes):
    def __init__(self, aggregation=AGGREGATION_NAIVE):
        super().__init__()
        self.aggregation = aggregation

    def generate_priori(self, dataset):
        if self.aggregation == AGGREGATION_DIAGONAL_EQUAL_PRIORI:
            self.priori = np.ones(self.num_classes) / self.num_classes
        else:
            super().generate_priori(dataset)

    def generate_cov_matrix(self, dataset):
        if self.aggregation == AGGREGATION_NAIVE:
            self.aggregate_naive(dataset)
        elif self.aggregation == AGGREGATION_POOL:
            self.aggregate_pool(dataset)
        elif self.aggregation == AGGREGATION_DIAGONAL_VARIANCE or self.aggregation == AGGREGATION_DIAGONAL_EQUAL_PRIORI:
            self.aggregate_diagonal_variance(dataset)
        else:
            raise Exception("Invalid aggregation")

    def aggregate_naive(self, dataset):
        x = dataset[:, :-1]
        self.cov_matrix = np.cov(x, rowvar=False)

    def aggregate_pool(self, dataset):
        super().generate_cov_matrix(dataset)
        self.cov_matrix *= self.priori.reshape(-1, 1, 1)
        self.cov_matrix = self.cov_matrix.sum(axis=0)

    def aggregate_diagonal_variance(self, dataset):
        self.aggregate_naive(dataset)
        variance = np.mean(np.diag(self.cov_matrix))
        self.cov_matrix = np.zeros(self.cov_matrix.shape)
        np.fill_diagonal(self.cov_matrix, variance)

    def cov_matrix_for_class(self, c):
        return self.cov_matrix
