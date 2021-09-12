import random
from knn.helpers import train_test_split, mean, standard_deviation
from knn.dataset import Dataset
from knn.knn import KNN
from knn.realization import Realization
from knn.scores import Scores
from knn.normalizer import Normalizer

# Import plotting modules, if they're available
try:
    from assignment7.plot_helper import plot_decision_surface, plot_regression_surface
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
        normalizer.fit(dataset)
        normalized_dataset = [normalizer.normalize(row[:-1]) + [row[-1]] for row in dataset]
    else:
        normalized_dataset = dataset

    realizations = list()
    for i in range(0, num_realizations):
        # Train the model
        training_set, test_set = train_test_split(normalized_dataset, ratio, shuffle=True)
        model.train(training_set)

        d = list()
        y = list()

        # Test the model
        for row in test_set:
            d.append(row[-1])
            y.append(model.predict(row[:-1]))

        # Caching realization values
        realization = Realization(training_set, test_set, Scores(d, y))
        print("Realization {}: {:.2f}%".format(i + 1, realization.scores.accuracy * 100))
        realizations.append(realization)

    # Accuracy Stats
    accuracies = list(map(lambda r: r.scores.accuracy, realizations))
    mean_accuracy = mean(accuracies)
    std_accuracy = standard_deviation(accuracies)
    print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

    # Realization whose accuracy is closest to the mean
    avg_realization = sorted(realizations, key=lambda r: abs(mean_accuracy - r.scores.accuracy))[0]

    print("Confusion matrix")
    avg_realization.scores.print_confusion_matrix()

    # Plot decision surface
    if len(dataset[0][:-1]) == 2 and plotting_available:
        # Set models with the "mean weights"
        plot_decision_surface(model,
                              normalized_dataset,
                              title="Superfície de Decisão",
                              xlabel="X1",
                              ylabel="X2")


# Dataset descriptors (lazy loaded)
# Artificial
artificial_dataset = Dataset("knn/datasets/artificial.csv")

# Iris
iris_dataset = Dataset("knn/datasets/iris.csv")

# Vertebral column
column_dataset = Dataset("knn/datasets/vertebral-column.csv")

# Best hyper parameter found using grid search with k-fold cross validation
hyper_parameters = {
    'artificial': (artificial_dataset, 7),
    'iris': (iris_dataset, 7),
    'column': (column_dataset, 7)
}

# Select best hyper parameters
# datasets = ['artificial', 'iris', 'column', 'dermatology', 'breast_cancer']
# for ds in datasets:
#     print(">>>>>>>>>>>>>> {}".format(ds))
#     dataset, _, _, _ = hyper_parameters['artificial']
# select_hyper_parameters(dermatology_dataset.load())
#     print("\n\n\n\n\n")

dataset, k = hyper_parameters['artificial']

split_ratio = 0.8
num_realizations = 20

print("Dataset: {}".format(dataset.filename))
model = KNN(k)
evaluate(model,
         dataset.load(),
         normalize=True,  # TODO: Test with and without normalization
         ratio=split_ratio,
         num_realizations=num_realizations)

print("Done!")
