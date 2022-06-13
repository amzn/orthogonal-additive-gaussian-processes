# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
# -

from oak.model_utils import oak_model
from oak.utils import (
    get_model_sufficient_statistics,
    get_prediction_component,
    initialize_kmeans_with_binary,
)


@pytest.mark.parametrize("binary_index", [[0, 1, 2], [0, 2], [1]])
@pytest.mark.parametrize("n_cluster", [1, 50])
def test_initialize_kmeans_with_binary(binary_index: list, n_cluster: int):
    np.random.seed(44)
    continuous_index = list(set(range(3)) - set(binary_index))
    N = 100
    dim = len(binary_index) + len(continuous_index)
    X = np.zeros((N, dim))
    for i in binary_index:
        print("shape of X ", X.shape)
        print("i = ", i)
        X[:, i] = np.random.binomial(1, 0.33, N)
    print("continuous_index", continuous_index)
    if len(continuous_index) > 0:
        for j in continuous_index:
            X[:, j] = np.random.normal(0, 4, N)
    else:
        continuous_index = None
    Z = initialize_kmeans_with_binary(X, binary_index, continuous_index, n_cluster)
    print(X)
    assert Z.shape == (n_cluster, dim)
    assert isinstance(Z, np.ndarray)


@pytest.mark.parametrize("share_var_across_orders", [True, False])
def test_get_prediction_component(share_var_across_orders: bool):
    # sum of predictions for all the functional components equals the final prediction
    np.random.seed(44)
    tf.random.set_seed(44)

    N = 2000
    X = np.random.normal(0, 1, (N, 3))
    y = (
        X[:, 0] ** 2 + X[:, 1] + X[:, 1] * X[:, 2] + np.random.normal(0, 0.01, (N,))
    ).reshape(-1, 1)

    oak = oak_model(
        num_inducing=50,
        max_interaction_depth=2,
        share_var_across_orders=share_var_across_orders,
    )
    oak.fit(X, y, optimise=False)
    oak.m.kernel.variances[0].assign(1e-16)
    oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)

    prediction_list = get_prediction_component(
        oak.m,
        oak.alpha,
        oak._transform_x(X),
        share_var_across_orders=share_var_across_orders,
    )
    out = np.zeros(y.shape[0])
    for i in range(len(prediction_list)):
        out += prediction_list[i].numpy()

    out_all = oak.m.predict_f(oak._transform_x(X))[0].numpy()[:, 0]
    print(f"variance 0 = {oak.m.kernel.variances[0]}")
    np.testing.assert_allclose(out, out_all)

