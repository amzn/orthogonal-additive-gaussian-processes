# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import gpflow
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from gpflow import set_trainable
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from scipy import io
from scipy.cluster.vq import kmeans
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib


matplotlib.rcParams.update({"font.size": 25})

data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data")
) + '/'

filenames = [
    data_path_prefix + "breast.mat",
    data_path_prefix + "pima.mat",
    data_path_prefix + "sonar.mat",
    data_path_prefix + "ionosphere.mat",
    data_path_prefix + "r_liver.mat",
    data_path_prefix + "r_heart.mat",
]
dataset_names = ["breast", "pima", "sonar", "ionosphere", "liver", "heart"]

np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})
np.random.seed(4)
tf.random.set_seed(4)

def inv_logit(x):
    jitter = 1e-3
    return tf.math.sigmoid(x) * (1 - 2 * jitter) + jitter


def main():
    """
    :param dataset_name: name of the dataset, should be one of the above dataset_names
    :param k: number of train-test fold, default to 5.
    :return: fit OAK model on the dataset, saves the model, the model predictive performance,
    and the plot on cumulative Sobol indices.
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--dataset_name", default="breast", type=str, help="dataset name"
    )
    args_parser.add_argument(
        "--k", type=int, default=5, help="k-fold train-test splits"
    )

    args, unknown = args_parser.parse_known_args()

    dataset_name, k = (
        args.dataset_name,
        args.k,
    )

    # save results to outputs folder
    output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/{dataset_name}/")
    )
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    np.random.seed(4)
    tf.random.set_seed(4)

    filename = filenames[dataset_names.index(dataset_name)]

    d = io.loadmat(filename)
    X, y = d["X"], d["y"]
    y = (y + 1) / 2
    idx = np.random.permutation(range(X.shape[0]))
    X = X[idx, :]
    y = y[idx]

    kf = KFold(n_splits=k)
    fold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # limit maximum number of interactions due to computation, sonar has 60 features therefore limiting it to 2
        # Sonar has ~60 features, truncating the maximum order of interaction to 2
        depth = 4 if dataset_name != "sonar" else 2
        oak = oak_model(max_interaction_depth=depth, num_inducing=200)
        oak.fit(X_train, y_train, optimise=False)
        data = (oak.m.data[0], y_train)
        t_start = time.time()

        Z = (
            kmeans(oak.m.data[0].numpy(), 200)[0]
            if X_train.shape[0] > 200
            else oak.m.data[0].numpy()
        )
        likelihood = gpflow.likelihoods.Bernoulli(invlink=inv_logit)

        oak.m = gpflow.models.SVGP(
            kernel=oak.m.kernel,
            likelihood=likelihood,
            inducing_variable=Z,
            whiten=True,
            q_diag=True,
        )

        set_trainable(oak.m.inducing_variable, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            oak.m.training_loss_closure(data),
            oak.m.trainable_variables,
            method="BFGS",
        )

        # test performance
        x_max, x_min = X_train.max(0), X_train.min(0)
        mu, var = oak.m.predict_f(oak._transform_x(np.clip(X_test, x_min, x_max)))
        prob = inv_logit(mu)
        classification_accuracy = np.sum(
            np.abs((prob > 0.5).numpy().astype(int)[:, 0] - y_test[:, 0])
        ) / len(y_test[:, 0])
        # clipping X_test to be in the range of training data to avoid nan error in calculating nll due to normalsing flow
        X_test_scaled = oak._transform_x(np.clip(X_test, x_min, x_max))
        nll = -oak.m.predict_log_density((X_test_scaled, y_test)).numpy().mean()

        print(f"fold {fold}, training dataset has size {X_train.shape}")
        print(
            f"oak test percentage classification error = {np.round(classification_accuracy, 4)}, "
            f"nll = {np.round(nll,4)}"
        )

        # calculate sobol
        oak.m.data = data
        Sobol = None
        try:
            oak.get_sobol()
            tuple_of_indices, normalised_sobols = (
                oak.tuple_of_indices,
                oak.normalised_sobols,
            )
            # aggregate sobol per order of interactions
            Sobol = np.zeros(len(tuple_of_indices[-1]))
            for i in range(len(tuple_of_indices)):
                Sobol[len(tuple_of_indices[i]) - 1] += normalised_sobols[i]
        except:
            # sobol calculation fails due to cholesky decomposition error
            print(f"Sobol calculation failed")
            pass
        print(f"sobol is {Sobol}")
        print(f"Computation took {time.time() - t_start:.1f} seconds\n")

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

        cumulative_sobol, auc_component = [], []
        order = np.argsort(normalised_sobols)[::-1]
        for n in order:
            # add predictions of the terms one by one ranked by their Sobol index
            y_pred_component += prediction_list[n].numpy()
            prob = inv_logit(y_pred_component)
            auc_component.append(roc_auc_score(y_test, prob.numpy()))
            cumulative_sobol.append(normalised_sobols[n])
        cumulative_sobol = np.cumsum(cumulative_sobol)

        # generate plots in Fig. 5 (\ref{fig:sobol_plots}) of paper
        plt.figure(figsize=(8, 4))
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(np.arange(len(order)), auc_component, "r", linewidth=4)
        ax2.plot(np.arange(len(order)), cumulative_sobol, "-.g", linewidth=4)

        ax1.set_xlabel("Number of Terms Added")
        ax1.set_ylabel("AUC", color="r")
        ax2.set_ylabel("Cumulative Sobol", color="g")

        plt.title(dataset_name)
        plt.tight_layout()
        plt.savefig(output_prefix + "/cumulative_sobol_%d.pdf" % fold)

        # aggregate sobol per order of interactions
        sobol_order = np.zeros(len(tuple_of_indices[-1]))
        for i in range(len(tuple_of_indices)):
            sobol_order[len(tuple_of_indices[i]) - 1] += normalised_sobols[i]
        # save learned model
        save_model(
            oak.m,
            filename=Path(output_prefix + f"/model_oak_%d" % fold),
        )
        # save model performance metrics
        np.savez(
            output_prefix + "/out_%d" % fold,
            normalised_sobols=normalised_sobols,
            classification_accuracy=classification_accuracy,
            nll=nll,
            sobol_order=sobol_order,
        )
        fold += 1
        
if __name__ == "__main__":
    main()
