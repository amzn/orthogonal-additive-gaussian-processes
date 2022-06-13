# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gpflow
import os
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from oak.model_utils import oak_model, load_model
from scipy import io
from pathlib import Path
# -


# this script use the saved model to plot top 5 important features (Sobol)
# in the decomposition
covariate_names = {}
covariate_names["Housing"] = [
    "crime",
    "zoned",
    "industrial",
    "river",
    "NOX",
    "rooms",
    "age",
    "empl. dist.",
    "highway acc.",
    "tax",
    "pupil ratio",
    "black pct",
    "low status pct",
]

covariate_names["concrete"] = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]
# no covariate name found for pumadyn
covariate_names["pumadyn"] = [f"input {i}" for i in range(8)]
covariate_names["autoMPG"] = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin",
]

covariate_names["breast"] = [
    "ClumpThickness",
    "CellSize",
    "CellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses",
]
covariate_names["pima"] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
covariate_names["sonar"] = [f"input {i}" for i in range(60)]
covariate_names["ionosphere"] = [f"input {i}" for i in range(32)]
covariate_names["liver"] = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks"]
covariate_names["heart"] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thelach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

data_path_prefix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data")
) + '/'

filenames = [
    data_path_prefix + "autompg.mat",
    data_path_prefix + "housing.mat",
    data_path_prefix + "r_concrete_1030.mat",
    data_path_prefix + "pumadyn8nh.mat",
    data_path_prefix + "breast.mat",
    data_path_prefix + "pima.mat",
    data_path_prefix + "sonar.mat",
    data_path_prefix + "ionosphere.mat",
    data_path_prefix + "r_liver.mat",
    data_path_prefix + "r_heart.mat",
]

dataset_names = [
    "autoMPG",
    "Housing",
    "concrete",
    "pumadyn",
    "breast",
    "pima",
    "sonar",
    "ionosphere",
    "liver",
    "heart",
]


def inv_logit(x):
    jitter = 1e-3
    return tf.math.sigmoid(x) * (1 - 2 * jitter) + jitter


def main():
    """
    :param dataset_name: name of the dataset, should be one of the above dataset_names
    :param fold: fold of the train-test splits to plot the model on, each fold has a model with different data
    :return: load model and plot the OAK decomposition
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--dataset_name", type=str, default="autoMPG", help="dataset name"
    )
    args_parser.add_argument(
        "--fold", type=int, default=0, help="Train-test split fold"
    )
    args, unknown = args_parser.parse_known_args()
    dataset_name, fold = args.dataset_name, args.fold

    filename = filenames[dataset_names.index(dataset_name)]

    print(f"dataset {dataset_name}\n")
    d = io.loadmat(filename)
    if dataset_name == "autoMPG":
        X, y = d["X"][:, 1:], d["X"][:, :1]
    else:
        X, y = d["X"], d["y"]
        if len(np.unique(y)) == 2:
            y = (y + 1) / 2

    # distinguish between regression and classification
    if len(np.unique(y)) > 2:
        oak = oak_model(max_interaction_depth=X.shape[1], lengthscale_bounds=None)
        oak.fit(X, y, optimise=False)

    else:
        depth = 4 if X.shape[1] < 40 else 2
        # number of inducing points was decided by the number of training instances (80% of the X)
        M = 200 if (X.shape[0] * 0.8) > 200 else int(X.shape[0] * 0.8)
        oak = oak_model(
            max_interaction_depth=depth, num_inducing=M, lengthscale_bounds=None
        )
        oak.fit(X, y, optimise=False)
        data = (oak.m.data[0], y)
        # Z is set to be a placeholder to load the values from saved model
        Z = (
            np.zeros((M, X.shape[1]))
            if M == 200
            else np.zeros((int(X.shape[0] * 0.8), X.shape[1]))
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
        oak.m.data = data

    # load the model and plot decomposition
    output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/{dataset_name}/")
    )
    load_model(
        oak.m,
        filename=Path(output_prefix + f"/model_oak_%d.npz" % fold),
        load_all_parameters=True,
    )
    oak.plot(
        top_n=5,
        semilogy=False,
        X_columns=covariate_names[dataset_name],
        save_fig=output_prefix + f"/decomposition",
    )


if __name__ == "__main__":
    main()
