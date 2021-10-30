from bayes.helpers import load_dataset
from bayes.bayes import GaussianBayes

import numpy as np

dataset = load_dataset('bayes/datasets/iris.csv')

# Univariate dataset
# size = 150
# x = np.arange(0, 150, 1)
# c = np.where(x < 75, 0, 1)
# dataset = np.stack((x, c)).T

model = GaussianBayes()
model.train(dataset)

count = 0
for point in dataset:
    expected = int(point[-1])
    predicted = model.predict(point[:-1])
    if expected == predicted:
        count += 1
print(count / 150)
