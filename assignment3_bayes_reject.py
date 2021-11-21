from bayes_reject.helpers import train_test_split
import numpy as np
from bayes_reject.dataset import Dataset
from bayes_reject.quadratic_bayes import QuadraticBayes
from bayes_reject.linear_bayes import LinearBayes, AGGREGATION_POOL, AGGREGATION_NAIVE,\
    AGGREGATION_DIAGONAL_VARIANCE, AGGREGATION_DIAGONAL_EQUAL_PRIORI
from bayes_reject.realization import Realization
from bayes_reject.scores import Scores
from bayes_reject.normalizer import Normalizer
import argparse

# Import plotting modules, if they're available
try:
    from bayes_reject.plot_helper import plot_decision_surface, plot_pdf
    import matplotlib.pyplot as plt

    plotting_available = True
except ModuleNotFoundError:
    plotting_available = False


def parse_args(dataset_params):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset",
                        help="The dataset to use. Choose iris, column or artificial.",
                        required=True,
                        choices=['artificial', 'breast-cancer', 'column', 'dermatology', 'iris'])
    parser.add_argument("-t",
                        "--discriminant",
                        help="The type of the discriminant to use",
                        choices=["d1", "d2", "d3", "d4", "d5"])
    args = parser.parse_args()

    if args.discriminant is None:
        print("Discriminant is required")
        exit(1)

    if args.discriminant == "d1":
        model = QuadraticBayes(wr=0)
    else:
        agg = {
            "d2": AGGREGATION_NAIVE,
            "d3": AGGREGATION_POOL,
            "d4": AGGREGATION_DIAGONAL_VARIANCE,
            "d5": AGGREGATION_DIAGONAL_EQUAL_PRIORI
        }
        if agg.get(args.discriminant) is None:
            print("Invalid linear aggregation")
            exit(1)
        model = LinearBayes(wr=0, aggregation=agg[args.discriminant])

    dataset = Dataset("bayes_reject/datasets/%s.csv" % args.dataset, klass=dataset_params[args.dataset])
    return dataset, model


def evaluate(model, dataset, normalize=True, ratio=0.8, num_realizations=20):
    if normalize:
        normalizer = Normalizer()
        normalizer.fit(dataset[:, :-1])
        normalized_dataset = [np.append(normalizer.normalize(row[:-1]), row[-1]) for row in dataset]
        normalized_dataset = np.array(normalized_dataset)
    else:
        normalized_dataset = dataset

    wrs = [0.48, 0.36, 0.24, 0.12, 0.04]
    wr_metrics = [list(), list()]

    for wr in wrs:
        print("Wr = {:.2f}".format(wr))

        realizations = list()

        for i in range(0, num_realizations):
            model.wr = wr

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
                                      model.priori,
                                      model.t,
                                      Scores(d, y))
            realizations.append(realization)

        # Accuracy Stats
        accuracies = np.array(list(map(lambda r: r.scores.accuracy, realizations)))
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print("Accuracy: {:.2f}% ± {:.2f}%".format(mean_accuracy * 100, std_accuracy * 100))

        # Rejection Stats
        rejections = np.array(list(map(lambda r: r.scores.rejection, realizations)))
        mean_rejection = np.mean(rejections)
        std_rejection = np.std(rejections)
        print("Rejection: {:.2f}% ± {:.2f}%".format(mean_rejection * 100, std_rejection * 100))

        wr_metrics[0].append(mean_rejection)
        wr_metrics[1].append(mean_accuracy)

        # Realization whose accuracy is closest to the mean
        avg_realization = sorted(realizations, key=lambda r: abs(mean_accuracy - r.scores.accuracy))[0]

        # print("Confusion matrix")
        # avg_realization.scores.print_confusion_matrix()

        # Plot decision surface
        # if dataset.shape[1] - 1 == 2 and plotting_available:
        #     # Set models with the "mean weights"
        #     model.means = avg_realization.means
        #     model.cov_matrix = avg_realization.cov_matrix
        #     model.t = avg_realization.t
        #     plot_decision_surface(model,
        #                           normalized_dataset,
        #                           title="Artificial",
        #                           xlabel="X1",
        #                           ylabel="X4",
        #                           legend={0: 'Class 0', 1: 'Class 1', -1: 'Rejected'})

    # AR Curve
    wr_metrics = np.array(wr_metrics)
    plt.plot(wr_metrics[0], wr_metrics[1], marker="s", markersize=10)
    plt.title("Curva AR")
    plt.ylabel("Taxa de Acerto")
    plt.xlabel("Taxa de Rejeição")
    plt.grid(which='both')
    plt.show()


# Dataset descriptors (lazy loaded)
dataset_params = {
    'artificial': None,
    'column': 2,
    'iris': 0,
}

# datasets = dict()
# for name, klass in dataset_params.items():
#     datasets[name] = Dataset("bayes_reject/datasets/%s.csv" % name, klass=klass)
# dataset = datasets['iris']

dataset, model = parse_args(dataset_params)

split_ratio = 0.6
num_realizations = 20

print("Dataset: {}".format(dataset.filename))
# model = QuadraticBayes(wr=0)
# model = LinearBayes(wr=0, aggregation=AGGREGATION_POOL)
evaluate(model=model,
         dataset=dataset.load(),
         normalize=True,
         ratio=split_ratio,
         num_realizations=num_realizations)

print("Done!")
