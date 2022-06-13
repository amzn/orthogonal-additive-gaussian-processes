# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from oak.input_measures import MOGMeasure
from oak.model_utils import oak_model


@pytest.mark.parametrize("interaction_depth", [2])
@pytest.mark.parametrize("use_sparsity_prior", [True, False])
@pytest.mark.parametrize("initialise_inducing_points", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("clip", [True, False])
def test_oak_model(
    interaction_depth: int,
    use_sparsity_prior: bool,
    initialise_inducing_points: bool,
    sparse: bool,
    clip: bool,
):
    np.random.seed(44)
    tf.random.set_seed(44)

    N = 100
    X = np.random.normal(0, 1, (N, 3))
    y = X[:, 0] ** 2 + X[:, 1] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.01, (N,))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y[:, None], test_size=0.2, random_state=42
    )

    oak = oak_model(
        num_inducing=50,
        max_interaction_depth=interaction_depth,
        use_sparsity_prior=use_sparsity_prior,
        sparse=sparse,
    )
    oak.fit(
        X_train,
        y_train,
        initialise_inducing_points=initialise_inducing_points,
        optimise=False,
    )

    y_pred = oak.predict(X_test, clip=clip)
    rss = mean_squared_error(y_pred, y_test[:, 0])
    # check the model is better than predicting using the mean prediction
    assert rss < mean_squared_error(
        y_test.mean() * np.ones(y_test[:, 0].shape), y_test[:, 0]
    )


@pytest.mark.parametrize("interaction_depth", [1, 2])
@pytest.mark.parametrize("use_sparsity_prior", [True, False])
def test_oak_model_with_binary_and_categorical_data(
    interaction_depth: int,
    use_sparsity_prior: bool,
):
    np.random.seed(44)
    tf.random.set_seed(44)

    N = 20
    x_cat = np.random.choice([0, 1, 2, 3], size=N, p=[0.2, 0.2, 0.3, 0.3])
    x_binary = np.random.choice([0, 1], size=N, p=[0.8, 0.2])
    x_cont = np.random.randn(N)
    X = np.vstack([x_binary, x_cat, x_cont]).T

    y = np.sin(X[:, 2]) + np.random.normal(0, 0.01, (N,))
    Y = y.reshape(-1, 1)

    oak = oak_model(
        binary_feature=[0],
        categorical_feature=[1],
        max_interaction_depth=interaction_depth,
        use_sparsity_prior=use_sparsity_prior,
    )
    oak.fit(X, Y, optimise=False)
    log_lik = (
        oak.m.log_marginal_likelihood()
    )  # check log likelihood calculation. This can reveal errors in shapes.
    assert not np.isnan(log_lik)


@pytest.fixture
def binary_5D_data():
    N = 3
    D = 5
    np.random.seed(42)
    X = np.random.randint(0, 2, N * D).reshape(N, D).astype(float)
    Y = np.random.randn(N, 1)
    return X, Y


@pytest.mark.parametrize(
    "binary_feature, categorical_feature, gmm_measure, empirical_measure",
    [
        [
            [0],
            [1],
            [0, 0, 2, 3, 0],
            [4],
        ],  # GMM with 2 and 3 clusters
        [[0], [1], None, [2, 3]],
    ],
)
def test_oak_model_creation(
    binary_5D_data,
    binary_feature,
    categorical_feature,
    gmm_measure,
    empirical_measure,
):
    X, Y = binary_5D_data
    oak = oak_model(
        num_inducing=3,
        binary_feature=binary_feature,
        categorical_feature=categorical_feature,
        gmm_measure=gmm_measure,
        empirical_measure=empirical_measure,
    )
    oak.fit(X, Y, optimise=False)


@pytest.mark.parametrize(
    "binary_feature, categorical_feature, gmm_measure, empirical_measure",
    [
        [[0], [1], None, None],
        [[0], [1, 3], None, None],
    ],
)
def test_oak_sobol_supported(
    binary_5D_data, binary_feature, categorical_feature, gmm_measure, empirical_measure
):
    X, Y = binary_5D_data
    # add random noise to input to make it continuous for continuous dimensions
    continuous_idx = list(set(np.arange(5)) - set(binary_feature + categorical_feature))
    X[:,continuous_idx] = X[:,continuous_idx] + np.random.normal(0, 1, (X.shape[0],len(continuous_idx)))
    
    oak = oak_model(
        binary_feature=binary_feature,
        categorical_feature=categorical_feature,
        gmm_measure=gmm_measure,
        empirical_measure=empirical_measure,
    )

    oak.fit(X, Y, optimise=False)

    sobol = oak.get_sobol()
    assert np.all(sobol >= 0)


@pytest.mark.parametrize(
    "gmm_measure",
    [[0, 0, 3, 0, 0]],
)
def test_oak_sobol_not_supported(binary_5D_data, gmm_measure):
    X, Y = binary_5D_data
    # add random noise to input to make it continuous 
    X = X + np.random.normal(0, 1, X.shape)

    oak = oak_model(
        gmm_measure=gmm_measure
    )
    oak.fit(X, Y, optimise=False)
    with pytest.raises(NotImplementedError):
        _ = oak.get_sobol()


# empirical_measure should throw for Sobol
# GMM should throw for Sobol
# empirical measure for discrete data only.. so should throw if specified of continuous
@pytest.mark.parametrize(
    "binary_feature, categorical_feature, gmm_measure, empirical_measure",
    [
        [[0, 1], [2], [0,0,0,2,0], [4]],  # empirical measure on binary
        [[0, 1], [2], None, [3, 4]],  # No GMM measure
    ],
)
def test_oak_good_model_creation_overlapping_indices(
    binary_5D_data,
    binary_feature,
    categorical_feature,
    gmm_measure,
    empirical_measure,
):
    X, Y = binary_5D_data
    oak = oak_model(
        binary_feature=binary_feature,
        categorical_feature=categorical_feature,
        gmm_measure=gmm_measure,
        empirical_measure=empirical_measure,
    )
    oak.fit(X, Y, optimise=False)


@pytest.mark.parametrize(
    "binary_feature, categorical_feature, gmm_measure, empirical_measure",
    [
        [[0, 1], [1], [0] * 5, [3]],  # overlapping binary & categorical
        [[0], [1], None, [0]],  # empirical measure on binary input
        [[0], [1], None, [1]],  # empirical measure on categorical input
        [
            [0],
            [1],
            [2, 0, 0, 0, 0],
            [2, 4],
        ],  # gmm specified on discrete input
    ],
)
def test_oak_illegal_model_creation_overlapping_indices(
    binary_5D_data,
    binary_feature,
    categorical_feature,
    gmm_measure,
    empirical_measure,
):
    X, Y = binary_5D_data
    oak = oak_model(
        binary_feature=binary_feature,
        categorical_feature=categorical_feature,
        gmm_measure=gmm_measure,
        empirical_measure=empirical_measure,
    )
    with pytest.raises(ValueError):
        oak.fit(X, Y, optimise=False)


def test_oak_gmm_applied_without_flows(binary_5D_data):
    np.random.seed(44)
    X, Y = binary_5D_data
    # add random noise to input to make it continuous for first and second dimension
    X[:,:-1] = X[:,:-1] + np.random.normal(0, 1, (X.shape[0],4))
    gmm_measure = [0, 0, 0, 0, 2]
    oak = oak_model(gmm_measure=gmm_measure)
    oak.fit(X, Y, optimise=False)
    assert oak.estimated_gmm_measures[:-1] == [None] * 4
    assert isinstance(oak.estimated_gmm_measures[-1], MOGMeasure)
    assert np.allclose(
        # check the means for the 3rd dimension is unchanged (0, 1)
        np.sort(oak.estimated_gmm_measures[-1].means), np.array([0, 1.0])
    )

    assert (
        oak.input_flows[-1] is None
    ), f"Flow applied on GMM measure input {oak.input_flows[-1]}"
    assert (
        np.array(oak.input_flows[:-1]) != None
    ).sum() == 4, f"Should have normalising flow for every continuous input without a GMM measure {oak.input_flows[:-1]}"

