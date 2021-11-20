import numpy as np


class Normalizer:
    def __init__(self):
        self.min_coefficients = None
        self.max_coefficients = None

    def fit(self, vector):
        self.min_coefficients = np.min(vector, axis=0)
        self.max_coefficients = np.max(vector, axis=0)

    def normalize(self, row):
        return (row - self.min_coefficients) / (self.max_coefficients - self.min_coefficients)

    def denormalize(self, row):
        return row * (self.max_coefficients - self.min_coefficients) + self.min_coefficients
