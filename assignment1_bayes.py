import random
from bayes.helpers import train_test_split
import numpy as np
from bayes.dataset import Dataset
from bayes.bayes import GaussianBayes
from bayes.realization import Realization
from bayes.scores import Scores
from bayes.normalizer import Normalizer

# Import plotting modules, if they're available
try:
    from bayes.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


# def select_hyper_parameters(dataset, k=5):
#     random.shuffle(dataset)
#     fold_size = int(len(dataset) / k)
#
#     hidden_layers = list(range(2, 70))
#     results = list()
#
#     for num_hidden in hidden_layers:
#         # for sigma in sigmas:
#         realizations = list()
#         for i in range(k):
#             test_start = i * fold_size
#             test_end = (i + 1) * fold_size
#
#             # Make training and test sets
#             training_set = list()
#             test_set = list()
#             for j in range(len(dataset)):
#                 if j < test_start or j >= test_end:
#                     training_set.append(dataset[j].copy())
#                 else:
#                     test_set.append(dataset[j].copy())
#
#             model = RBF(num_hidden=num_hidden, regression=False)
#             model.train(training_set)
#
#             d = list()
#             y = list()
#
#             # Validate the model
#             for row in test_set:
#                 d.append(row[-1])
#                 y.append(model.predict(row[:-1]))
#
#             realization = Realization(training_set, test_set, None, Scores(d, y), None)
#             realizations.append(realization)
#
#         accuracies = list(map(lambda r: r.scores.accuracy, realizations))
#         mean_accuracy = mean(accuracies)
#         print(
#             "Hidden: {}     Accuracy: {:.2f}%".format(
#                 num_hidden, mean_accuracy * 100
#             )
#         )
#
#         results.append((num_hidden, mean_accuracy))
#
#     results = sorted(results, key=lambda r: r[1], reverse=True)
#     best_hyper_parameters = results[0]
#     print("\n\n>>> Best hyper parameters:")
#     print("Hidden: {}     Accuracy: {:.2f}%".format(best_hyper_parameters[0], best_hyper_parameters[1] * 100))


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


# Dataset descriptors (lazy loaded)
# Iris
iris_dataset = Dataset("bayes/datasets/iris.csv")

# Best hyper parameter found using grid search with k-fold cross validation
# hyper_parameters = {
#     'artificial': (artificial_dataset, False, 10),
#     'iris': (iris_dataset, False, 20),
#     'column': (column_dataset, False, 40),
#     'dermatology': (dermatology_dataset, False, 49),
#     'breast_cancer': (breast_cancer_dataset, False, 2),
#     'artificial_regression': (artificial_regression_dataset, True, 8),
#     'abalone': (abalone_dataset, True, 20),
#     'car': (car_dataset, True, 30),
#     'motor': (motor_dataset, True, 20),
# }

# Select best hyper parameters
# datasets = ['artificial', 'iris', 'column', 'dermatology', 'breast_cancer']
# for ds in datasets:
#     print(">>>>>>>>>>>>>> {}".format(ds))
#     dataset, _, _, _ = hyper_parameters['artificial']
# select_hyper_parameters(dermatology_dataset.load())
#     print("\n\n\n\n\n")

# dataset, regression, hidden_layers = hyper_parameters['artificial']
dataset = iris_dataset

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
