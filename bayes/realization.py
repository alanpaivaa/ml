class Realization:
    def __init__(self, training_set=None, test_set=None, means=None, cov_matrix=None, scores=None):
        self.training_set = training_set
        self.test_set = test_set
        self.means = means
        self.cov_matrix = cov_matrix
        self.scores = scores
