from bayes_linear.helpers import train_test_split
import numpy as np
from bayes_linear.dataset import Dataset
from bayes_linear.bayes import GaussianBayes
from bayes_linear.realization import Realization
from bayes.scores import Scores
from bayes_linear.normalizer import Normalizer
import argparse

# Import plotting modules, if they're available
try:
    from bayes_linear.plot_helper import plot_decision_surface, plot_pdf
    import matplotlib.pyplot as plt

    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset",
                        help="The dataset to use. Choose iris, column or artificial.",
                        required=True,
                        choices=['artificial', 'breast-cancer', 'column', 'dermatology', 'iris'])
    args = parser.parse_args()
    return args.dataset


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
                              title="Íris",
                              xlabel="X1",
                              ylabel="X4", legend={0: 'Setosa', 1: 'Versicolor', 2: 'Virgínica'})


# Generate artificial dataset
# generate_artificial_dataset(plotting_available)

# Dataset descriptors (lazy loaded)
artificial_dataset = Dataset("bayes_linear/datasets/artificial.csv")
breast_cancer_dataset = Dataset("bayes_linear/datasets/breast-cancer.csv")
column_dataset = Dataset("bayes_linear/datasets/column.csv")
dermatology_dataset = Dataset("bayes_linear/datasets/dermatology.csv")
iris_dataset = Dataset("bayes_linear/datasets/iris.csv")

datasets = {
    'artificial': artificial_dataset,
    'breast-cancer': breast_cancer_dataset,
    'column': column_dataset,
    'dermatology': dermatology_dataset,
    'iris': iris_dataset
}

dataset = iris_dataset
# dataset = datasets[parse_args()]

# Plot PDF for a column of dataset
# col = 0
# plot_pdf(dataset.load(), col,
#          title="PDF - Coluna",
#          x_label="X%d" % (col + 1),
#          y_label="Densidade",
#          legend=None)

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
