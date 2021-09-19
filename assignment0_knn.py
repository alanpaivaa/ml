import random
from knn.helpers import train_test_split, mean, standard_deviation
from knn.dataset import Dataset
from knn.knn import KNN
from knn.dmc import DMC
from knn.realization import Realization
from knn.scores import Scores
from knn.normalizer import Normalizer
import argparse

# Import plotting modules, if they're available
try:
    from knn.plot_helper import plot_decision_surface
    import matplotlib.pyplot as plt
    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--model",
                        help="The model to use. Choose knn or dmc.",
                        required=True,
                        choices=['knn', 'dmc'])
    parser.add_argument("-d",
                        "--dataset",
                        help="The dataset to use. Choose iris, column or artificial.",
                        required=True,
                        choices=['iris', 'column', 'artificial'])
    args = parser.parse_args()
    return args.model, args.dataset


def select_hyper_parameters(model_class, dataset, normalize, num_folds=5):
    if normalize:
        normalizer = Normalizer()
        normalizer.fit(dataset)
        normalized_dataset = [normalizer.normalize(row[:-1]) + [row[-1]] for row in dataset]
    else:
        normalized_dataset = dataset

    random.shuffle(normalized_dataset)
    fold_size = int(len(normalized_dataset) / num_folds)

    results = list()
    k_range = range(1, 50)

    for k in k_range:
        realizations = list()
        for i in range(num_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size

            # Make training and test sets
            training_set = list()
            test_set = list()
            for j in range(len(normalized_dataset)):
                if j < test_start or j >= test_end:
                    training_set.append(normalized_dataset[j].copy())
                else:
                    test_set.append(normalized_dataset[j].copy())

            kwargs = {'k': k}
            model = model_class(**kwargs)
            model.train(training_set)

            d = list()
            y = list()

            # Validate the model
            for row in test_set:
                d.append(row[-1])
                y.append(model.predict(row[:-1]))

            realization = Realization(training_set, test_set, Scores(d, y))
            realizations.append(realization)

        accuracies = list(map(lambda r: r.scores.accuracy, realizations))
        mean_accuracy = mean(accuracies)
        print(
            "K: {}     Accuracy: {:.2f}%".format(
                k, mean_accuracy * 100
            )
        )

        results.append((k, mean_accuracy))

    results = sorted(results, key=lambda r: r[1], reverse=True)
    best_hyper_parameters = results[0]
    print("\n\n>>> Best hyper parameters:")
    print("K: {}     Accuracy: {:.2f}%".format(best_hyper_parameters[0], best_hyper_parameters[1] * 100))


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


model_name = 'dmc'
dataset_name = 'artificial'
normalize = False
# model_name, dataset_name = parse_args()

models = {
    'knn': KNN,
    'dmc': DMC
}

# Dataset descriptor, lazy loaded
dataset = Dataset("knn/datasets/{}.csv".format(dataset_name))
print("Dataset: {}".format(dataset.filename))

# Optimization
# select_hyper_parameters(model_class=models[model_name],
#                         dataset=dataset.load(),
#                         normalize=normalize,
#                         num_folds=5)

# Best hyper parameter found using grid search with k-fold cross validation
if normalize:  # Normalized
    hyper_parameters = {
        ('knn', 'artificial'): {'k': 1},
        ('knn', 'iris'): {'k': 7},
        ('knn', 'column'): {'k': 21},
        ('dmc', 'artificial'): dict(),
        ('dmc', 'iris'): dict(),
        ('dmc', 'column'): dict()
    }
else:  # Un-normalized
    hyper_parameters = {
        ('knn', 'artificial'): {'k': 1},
        ('knn', 'iris'): {'k': 16},
        ('knn', 'column'): {'k': 4},
        ('dmc', 'artificial'): dict(),
        ('dmc', 'iris'): dict(),
        ('dmc', 'column'): dict()
    }

model_kwargs = hyper_parameters[(model_name, dataset_name)]
model = models[model_name](**model_kwargs)

# Evaluation
split_ratio = 0.8
num_realizations = 20
evaluate(model,
         dataset.load(),
         normalize=normalize,
         ratio=split_ratio,
         num_realizations=num_realizations)

print("Done!")
