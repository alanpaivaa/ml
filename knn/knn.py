import math


class KNN:
    def __init__(self, k):
        self.training_set = None
        self.k = k

    def train(self, training_set):
        self.training_set = training_set

    def euclidean_distance(self, x, y):
        summation = 0
        count = min(len(x), len(y))
        for i in range(count):
            summation += (x[i] - y[i]) ** 2
        return math.sqrt(summation)

    def get_neighbors(self, point):
        distances = []
        for i in range(len(self.training_set)):
            distance = self.euclidean_distance(self.training_set[i][:-1], point)
            distances.append((i, distance))
        distances.sort(key=lambda x: x[1])
        indexes = list(map(lambda x: x[0], distances[:self.k]))
        result = list(map(lambda index: self.training_set[index], indexes))
        return result

    def max_class(self, neighbors):
        count = dict()
        for row in neighbors:
            klass = row[-1]
            if klass not in count:
                count[klass] = 0
            count[klass] += 1
        return max(count, key=lambda key: count[key])

    def predict(self, point):
        # Point in test SHOULD NOT include the class value
        assert(len(self.training_set[0]) - 1 == len(point))
        neighbors = self.get_neighbors(point)
        prediction = self.max_class(neighbors)
        return prediction