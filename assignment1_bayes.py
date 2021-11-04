import random
from bayes.helpers import train_test_split
import numpy as np
from bayes.dataset import Dataset, generate_artificial_dataset
from bayes.bayes import GaussianBayes
from bayes.realization import Realization
from bayes.scores import Scores
from bayes.normalizer import Normalizer

# Import plotting modules, if they're available
try:
    from bayes.plot_helper import plot_decision_surface, plot_pdf
    import matplotlib.pyplot as plt

    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def evaluate(model, dataset, normalize=True, ratio=0.8, num_realizations=20):
    if normalize:
        normalizer = Normalizer()
        normalizer.fit(dataset[:, :-1])
        normalized_dataset = [np.append(normalizer.normalize(row[:-1]), row[-1]) for row in dataset]
        normalized_dataset = np.array(normalized_dataset)
    else:
        normalized_dataset = dataset

    realizations = list()
    for i in range(0, num_realizations):
        # Train the model
        training_set, test_set = train_test_split(normalized_dataset, ratio, shuffle=True)
        model.train(training_set)

        d = np.array([]).astype('int')
        y = np.array([]).astype('int')

        # Test the model
        for row in test_set:
            d = np.append(d, int(row[-1]))
            y = np.append(y, int(model.predict(row[:-1])))

        # Caching realization values
        realization = Realization(training_set,
                                  test_set,
                                  model.means,
                                  model.cov_matrix,
                                  Scores(d, y))
        print("Realization {}: {:.2f}%".format(i + 1, realization.scores.accuracy * 100))
        realizations.append(realization)

    # Accuracy Stats
    accuracies = np.array(list(map(lambda r: r.scores.accuracy, realizations)))
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    # Realization whose accuracy is closest to the mean
    avg_realization = sorted(realizations, key=lambda r: abs(mean_accuracy - r.scores.accuracy))[0]

    print("Confusion matrix")
    avg_realization.scores.print_confusion_matrix()

    # Plot decision surface
    if dataset.shape[1] - 1 == 2 and plotting_available:
        # Set models with the "mean weights"
        model.means = avg_realization.means
        model.cov_matrix = avg_realization.cov_matrix
        plot_decision_surface(model,
                              normalized_dataset,
                              title="Superfície de Decisão",
                              xlabel="X1",
                              ylabel="X2")


# Generate artificial dataset
# generate_artificial_dataset(plotting_available)

# Dataset descriptors (lazy loaded)
iris_dataset = Dataset("bayes/datasets/iris.csv")
column_dataset = Dataset("bayes/datasets/column.csv")
artificial_dataset = Dataset("bayes/datasets/artificial.csv")
dermatology_dataset = Dataset("bayes/datasets/dermatology.csv")
breast_cancer_dataset = Dataset("bayes/datasets/breast-cancer.csv")
dataset = iris_dataset

# Plot PDF for a column of dataset
# plot_pdf(dataset.load(), 1)

split_ratio = 0.8
num_realizations = 20

print("Dataset: {}".format(dataset.filename))
model = GaussianBayes()
evaluate(model,
         dataset.load(),
         normalize=True,
         ratio=split_ratio,
         num_realizations=num_realizations)

print("Done!")
