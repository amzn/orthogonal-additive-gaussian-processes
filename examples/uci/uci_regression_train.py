# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
    
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path
matplotlib.rcParams.update({"font.size": 25})

# +
# data from repo: https://github.com/duvenaud/additive-gps/blob/master/data/regression/
# this script is for experiments in Sec 5.1 for regression problems in the paper
data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data")
) + '/'

filenames = [
    data_path_prefix + "autompg.mat",
    data_path_prefix + "housing.mat",
    data_path_prefix + "r_concrete_1030.mat",
    data_path_prefix + "pumadyn8nh.mat",
]
dataset_names = ["autoMPG", "Housing", "concrete", "pumadyn"]
np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})


# -


def main():
    """
    :param dataset_name: name of the dataset, should be one of the above dataset_names
    :param k: number of train-test fold, default to 5.
    :return: fit OAK model on the dataset, saves the model, the model predictive performance,
    and the plot on cumulative Sobol indices.
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--dataset_name", default="autoMPG", type=str, help="dataset name"
    )
    args_parser.add_argument(
        "--k", type=int, default=5, help="k-fold train-test splits"
    )

    args, unknown = args_parser.parse_known_args()
    dataset_name, k = args.dataset_name, args.k

    # save results to outputs folder
    output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/{dataset_name}/")
    )
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    np.random.seed(4)
    tf.random.set_seed(4)
    filename = filenames[dataset_names.index(dataset_name)]

    print(f"dataset {dataset_name}\n")
    d = io.loadmat(filename)
    if dataset_name == "autoMPG":
        # for autoMPG dataset, the first column is the response y
        X, y = d["X"][:, 1:], d["X"][:, :1]
    else:
        X, y = d["X"], d["y"]

    idx = np.random.permutation(range(X.shape[0]))

    X = X[idx, :]
    y = y[idx]
    kf = KFold(n_splits=k)
    fold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        oak = oak_model(
            max_interaction_depth=X.shape[1],
            num_inducing=500,
        )
        oak.fit(X_train, y_train)

        # test performance
        x_max, x_min = X_train.max(0), X_train.min(0)
        y_pred = oak.predict(np.clip(X_test, x_min, x_max))
        rss = ((y_pred - y_test[:, 0]) ** 2).mean()
        tss = (
            (y_test[:, 0] - y_test[:, 0].mean() * np.ones(y_test[:, 0].shape)) ** 2
        ).mean()
        r2 = 1 - rss / tss
        rmse = np.sqrt(rss)

        # calculate sobol
        oak.get_sobol()
        tuple_of_indices, normalised_sobols = (
            oak.tuple_of_indices,
            oak.normalised_sobols,
        )

        # cumulative Sobol as we add terms one by one ranked by their Sobol
        x_max, x_min = X_train.max(0), X_train.min(0)
        XT = oak._transform_x(np.clip(X_test, x_min, x_max))
        oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)
        # get the predicted y for all the kernel components
        prediction_list = get_prediction_component(
            oak.m,
            oak.alpha,
            XT,
        )
        # predicted y for the constant kernel
        constant_term = oak.alpha.numpy().sum() * oak.m.kernel.variances[0].numpy()
        print(f"constant_term = {constant_term}")
        y_pred_component = np.ones(y_test.shape[0]) * constant_term

        cumulative_sobol, rmse_component = [], []
        order = np.argsort(normalised_sobols)[::-1]
        for n in order:
            # add predictions of the terms one by one ranked by their Sobol index
            y_pred_component += prediction_list[n].numpy()
            y_pred_component_transformed = oak.scaler_y.inverse_transform(
                y_pred_component.reshape(-1, 1)
            )
            error_component = np.sqrt(
                ((y_pred_component_transformed - y_test) ** 2).mean()
            )
            rmse_component.append(error_component)
            cumulative_sobol.append(normalised_sobols[n])
        cumulative_sobol = np.cumsum(cumulative_sobol)

        # sanity check that predictions by summing over the components is equal
        # to the prediction of the OAK model
        np.testing.assert_allclose(y_pred_component_transformed[:, 0], y_pred)
        # generate plots in Fig. 5 (\ref{fig:sobol_plots}) of paper
        plt.figure(figsize=(8, 4))
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(np.arange(len(order)), rmse_component, "r", linewidth=4)
        ax2.plot(np.arange(len(order)), cumulative_sobol, "-.g", linewidth=4)

        ax1.set_xlabel("Number of Terms Added")
        ax1.set_ylabel("RMSE", color="r")
        ax2.set_ylabel("Cumulative Sobol", color="g")

        plt.title(dataset_name)
        plt.tight_layout()
        plt.savefig(output_prefix + "/cumulative_sobol_%d.pdf" % fold)

        # aggregate sobol per order of interactions
        sobol_order = np.zeros(len(tuple_of_indices[-1]))
        for i in range(len(tuple_of_indices)):
            sobol_order[len(tuple_of_indices[i]) - 1] += normalised_sobols[i]

        nll = (
            -oak.m.predict_log_density(
                (
                    oak._transform_x(np.clip(X_test, x_min, x_max)),
                    oak.scaler_y.transform(y_test),
                )
            )
            .numpy()
            .mean()
        )
        # printing
        print(f"fold {fold}, training dataset has size {X_train.shape}")
        print(f"sobol per interaction order is {sobol_order}")
        print(f"oak test rmse = {rmse}, r2 = {r2}\n")
        print(f"RBF test nll = {np.round(nll, 4)}\n")
        # save learned model
        save_model(
            oak.m,
            filename=Path(output_prefix + "/model_oak_%d" % fold),
        )
        # save model performance metrics
        np.savez(
            output_prefix + "/out_%d" % fold,
            cumulative_sobol=cumulative_sobol,
            order=order,
            rmse=rmse,
            nll=nll,
            sobol_order=sobol_order,
        )
        fold += 1


if __name__ == "__main__":
    main()
