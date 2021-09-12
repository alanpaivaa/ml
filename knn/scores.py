class Scores:
    def __init__(self, classes, predicted):
        self.classes = classes
        self.predicted = predicted

        # Count the number of unique values in both lists
        self.num_classes = len(set(self.classes + self.predicted))
        self.accuracy = None
        self.confusion_matrix = None

        self.compute_accuracy()
        self.compute_confusion_matrix()

    def compute_accuracy(self):
        correct = 0
        for i in range(len(self.predicted)):
            if self.classes[i] == self.predicted[i]:
                correct += 1
        self.accuracy = (correct / len(self.predicted))

    def compute_confusion_matrix(self):
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        for i in range(len(self.classes)):
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
        for i in range(len(self.confusion_matrix)):
            str_count = " ".join(map(lambda x: str(x).rjust(d, " "), self.confusion_matrix[i]))
            line = "\n{} {}".format(i, str_count)
            result += line
        print(result)
