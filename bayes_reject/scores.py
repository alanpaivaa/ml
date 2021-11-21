import numpy as np


class Scores:
    def __init__(self, classes, predicted):
        self.classes = classes
        self.predicted = predicted
        self.non_rejected_predicted = None
        self.non_rejected_classes = None
        self.confusion_matrix = None
        self.num_classes = None
        self.rejection = None
        self.accuracy = None

        self.compute_num_classes()
        self.compute_rejection()
        self.compute_accuracy()
        self.compute_confusion_matrix()

    def compute_num_classes(self):
        all_classes = np.unique(np.concatenate((self.classes, self.predicted)))
        self.num_classes = all_classes.shape[0]
        if -1 in all_classes:
            self.num_classes -= 1

    def compute_rejection(self):
        num_rejected = (self.predicted < 0).sum()
        self.rejection = num_rejected / self.predicted.shape[0]

    def compute_accuracy(self):
        non_rejected_indexes = self.predicted >= 0
        self.non_rejected_predicted = self.predicted[non_rejected_indexes]
        self.non_rejected_classes = self.classes[non_rejected_indexes]
        self.accuracy = np.sum(self.non_rejected_classes == self.non_rejected_predicted) / self.non_rejected_classes.shape[0]

    def compute_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes)).astype('int')
        for i in range(self.non_rejected_classes.shape[0]):
            self.confusion_matrix[self.non_rejected_predicted[i]][self.non_rejected_classes[i]] += 1

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
