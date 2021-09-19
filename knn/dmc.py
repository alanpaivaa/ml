from knn.knn import KNN


class DMC(KNN):
    def __init__(self):
        super().__init__(1)

    @staticmethod
    def get_centroids(points):
        # Group rows by class
        groups = dict()
        for row in points:
            klass = row[-1]
            if groups.get(klass) is None:
                groups[klass] = list()
            groups[klass].append(row)

        # Create centroids
        centroids = list()
        for klass in groups:
            centroid = list()
            group = groups[klass]
            size = len(group[0]) - 1  # Last element is the class

            # Calculate mean for index i
            for i in range(size):
                acc = 0
                for row in group:
                    acc += row[i]
                centroid.append(acc / len(group))

            # Add class as last index of centroid
            centroid.append(klass)
            centroids.append(centroid)

        return centroids

    def train(self, training_set):
        self.training_set = self.get_centroids(training_set)
