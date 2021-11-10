import numpy as np


class Scores:
    def __init__(self, classes, predicted):
        self.classes = classes
        self.predicted = predicted
        self.confusion_matrix = None
        self.num_classes = np.unique(np.concatenate((self.classes, self.predicted))).shape[0]
        self.accuracy = np.sum(self.classes == self.predicted) / self.classes.shape[0]
        self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes)).astype('int')
        for i in range(self.classes.shape[0]):
            self.confusion_matrix[self.predicted[i]][self.classes[i]] += 1

    def max_digits_count(self):
        count = 0
        for i in range(len(self.confusion_matrix)):
            for number in self.confusion_matrix[i]:
                count = max(count, len(str(number)))
        return count

    def print_confusion_matrix(self):
        d = self.max_digits_count()
        result = "  " + " ".join([str(i).rjust(d, " ") for i in range(len(self.confusion_matrix))])
        for i in range(self.confusion_matrix.shape[0]):
            str_count = " ".join(map(lambda x: str(x).rjust(d, " "), self.confusion_matrix[i]))
            line = "\n{} {}".format(i, str_count)
            result += line
        print(result)
