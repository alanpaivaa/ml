class Normalizer:
    def __init__(self):
        self.coefficients = None

    def fit_col(self, vector, col):
        x_min = float('inf')
        x_max = float('-inf')
        for i in range(len(vector)):
            x_min = min(x_min, vector[i][col])
            x_max = max(x_max, vector[i][col])
        self.coefficients[col] = (x_min, x_max)

    def fit(self, vector):
        self.coefficients = [None for _ in range(len(vector[0]))]

        for i in range(len(vector[0])):
            self.fit_col(vector, i)

    def normalize(self, row):
        normalized = [0 for _ in range(len(row))]
        for i in range(len(row)):
            x_min, x_max = self.coefficients[i]
            normalized[i] = (row[i] - x_min) / (x_max - x_min)
        return normalized

    def denormalize(self, row):
        denormalized = [0 for _ in range(len(row))]
        for i in range(len(row)):
            x_min, x_max = self.coefficients[i]
            denormalized[i] = row[i] * (x_max - x_min) + x_min
        return denormalized
