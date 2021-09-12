from knn.helpers import load_dataset


class Dataset:
    def __init__(self, filename, regression=False, encoding=None, features=None):
        self.regression = regression
        self.filename = filename
        self.encoding = encoding
        self.features = features

    def generate_encoding(self, dataset):
        if type(dataset[0][-1]) != str:
            return

        self.encoding = dict()
        encoding_set = set()
        for row in dataset:
            if row[-1] not in encoding_set:
                self.encoding[row[-1]] = len(encoding_set)
                encoding_set.add(row[-1])

    def encoded_classes(self, dataset):
        # For values not contained in the encoding dict, we use the value max + 1
        unknown = max(self.encoding.values()) + 1
        for row in dataset:
            klass = row[-1]  # Last element of the array is the class
            if self.encoding.get(klass) is not None:
                row[-1] = self.encoding[row[-1]]
            else:
                row[-1] = unknown
        return dataset

    @staticmethod
    def set_last_column_int(dataset):
        for row in dataset:
            row[-1] = int(row[-1])

    def load(self):
        dataset = load_dataset(self.filename, self.regression)

        # Selecting only the needed columns
        if self.features is not None:
            for i in range(len(dataset)):
                new_row = list()
                for f in self.features:
                    new_row.append(dataset[i][f])
                new_row.append(dataset[i][-1])
                dataset[i] = new_row

        if not self.regression:
            if self.encoding is None:
                self.generate_encoding(dataset)

            if self.encoding is not None:
                dataset = self.encoded_classes(dataset)
            self.set_last_column_int(dataset)
        return dataset
