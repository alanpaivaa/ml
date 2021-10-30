from bayes.helpers import load_dataset


class Dataset:
    def __init__(self, filename, encoding=None, features=None):
        self.filename = filename
        self.encoding = encoding
        self.features = features

    def load(self):
        dataset = load_dataset(self.filename)

        # Selecting only the needed columns
        if self.features is not None:
            dataset = dataset[:, self.features + [-1]]

        return dataset
